import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

from Tokenizer import Tokenizer

# Configuration
local_dir = "puzzles_dicts_tokens"
shard_size = int(1e8)       # 100M tokens per shard
mode = "biased_1_1"           # shuffled / sorted / staged_n (n: number of stages) / biased_b_n (b: bias, n: number of iterations)
epochs = 4
csv_path = "Puzzles/Benchmark/puzzle_bench.csv"

# Create the cache directory if it doesn't exist
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Load data
og_df = pd.read_csv(csv_path)

# Initialize the tokenizer
enc = Tokenizer()

def dict_to_string(row):
    # Convert a DataFrame row to a string representation of a dictionary
    return '{' + ','.join(f'"{key}":"{value}"' for key, value in row.items() if pd.notna(value)) + '}'

def tokenize(row):
    # Tokenize the string representation of the dictionary
    text = dict_to_string(row)
    tokens = [0]  # Start with the end of text token
    tokens.extend(enc.encode(text))
    tokens_np = np.array(tokens, dtype=np.uint16)
    return tokens_np

def process_row(row):
    # Wrapper function to tokenize a single row
    return tokenize(row[1])

def write_datafile(filename, tokens_np):
    # Save the tokenized data to a .npy file
    np.save(filename, tokens_np)

def shuffle_data(data):
    """Shuffles the entire DataFrame."""
    return data.sample(frac=1).reset_index(drop=True)

def sort_data(data):
    """Sorts the DataFrame based on the 'Rating' field."""
    return data.sort_values(by='Rating').reset_index(drop=True)

def staged_data(data, n):
    """Sorts the DataFrame based on 'Rating', splits it into `n` parts, shuffles each part, and then concatenates them."""
    sorted_data = sort_data(data)
    chunks = np.array_split(sorted_data, n)  # Split into n equal parts
    shuffled_chunks = [chunk.sample(frac=1).reset_index(drop=True) for chunk in chunks]
    return pd.concat(shuffled_chunks).reset_index(drop=True)

def biased_sort_data(data, bias_strength, iterations):
    """Sorts the DataFrame using a biased sort over multiple iterations."""
    positions = np.zeros(len(data))
    
    for _ in range(iterations):
        shuffled_indices = data.index.to_list()
        random.shuffle(shuffled_indices)
        
        for i, idx in enumerate(shuffled_indices):
            biased_value = data.loc[idx, 'Rating'] + bias_strength * random.uniform(-data.loc[idx, 'Rating'], data.loc[idx, 'Rating'])
            positions[idx] += biased_value
    
    sorted_indices = np.argsort(positions)
    sorted_data = data.iloc[sorted_indices].reset_index(drop=True)
    
    return sorted_data

if __name__ == "__main__":

    df = pd.DataFrame([])
    for i in range(epochs):
        if mode == "shuffled":
            df = pd.concat([df, shuffle_data(og_df)])
        elif mode == "sorted":
            df = pd.concat([df, sort_data(og_df)])
        elif "staged" in mode:
            if len(mode.split('_')) != 2:
                print(f"Error> number of arguments for staged should be 2: {mode}")
                exit()
            n = int(mode.split('_')[1])
            if isinstance(n, int):
                df = pd.concat([df, staged_data(og_df, n)])
            else:
                print(f"Error> invalid n: {n}")
                exit()
        elif "biased" in mode:
            if len(mode.split('_')) != 3:
                print(f"Error> number of arguments for biased should be 3: {mode}")
                exit()
            b = float(mode.split('_')[1])
            n = int(mode.split('_')[2])
            if isinstance(n, int) and isinstance(b, float):
                df = pd.concat([df, biased_sort_data(og_df, b, n)])
            else:
                print(f"Error> invalid n or b: {n}, {b}")
                exit()
        else:
            print(f"Error> invalid mode: {mode}")
            exit()

    # Tokenize all rows and write output shards
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # Preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        # Use `imap` to process rows in parallel
        for tokens in pool.imap(process_row, df.iterrows(), chunksize=16):
            tokens_length = len(tokens)
            # Is there enough space in the current shard for the new tokens?
            if token_count + tokens_length <= shard_size:
                # Append tokens to the current shard
                all_tokens_np[token_count:token_count + tokens_length] = tokens
                token_count += tokens_length
                # Update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(tokens_length)
            else:
                # Write the current shard and start a new one
                filename = os.path.join(DATA_CACHE_DIR, f"puzzles_train_{shard_index:06d}.npy")
                # Write the remainder of the tokens to the current shard
                remainder = shard_size - token_count
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np[:token_count + remainder])
                shard_index += 1
                progress_bar = None
                # Populate the next shard with the leftovers of the current doc
                all_tokens_np[:tokens_length - remainder] = tokens[remainder:]
                token_count = tokens_length - remainder

        # Write any remaining tokens as the last shard
        if token_count != 0:
            filename = os.path.join(DATA_CACHE_DIR, f"puzzles_train_{shard_index:06d}.npy")
            write_datafile(filename, all_tokens_np[:token_count])
