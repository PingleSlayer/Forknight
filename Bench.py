import pandas as pd
import json
import re
import time
import math
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch.nn import functional as F


def calculate_elo(current_elo, puzzle_rating, score, K=16):
    expected_score = 1 / (1 + 10 ** ((puzzle_rating - current_elo) / 400))
    new_elo = current_elo + K * (score - expected_score)
    return new_elo

def evaluate_puzzle(generated_moves, correct_moves):
    if not generated_moves:
        return 0
    generated_sequence = generated_moves.split()
    correct_sequence = correct_moves.split()
    min_length = min(len(generated_sequence), len(correct_sequence))

    correct_count = 0    
    for i in range(min_length):
        if generated_sequence[i] == correct_sequence[i]:
            correct_count += 1
        else:
            break
    score = correct_count / len(correct_sequence)
    return score

def do_puzzle(fen, device, device_type, enc, model, ddp_rank):
    prompt = f'{{"FEN":"{fen}","'
    tokens = enc.encode(prompt)
    max_length = 128 + len(tokens)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(1, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)

    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(xgen) # (B, T, vocab_size)

            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)

            if xcol.item() == 0:
                tokens = xgen[0, :max_length].tolist()
                output = enc.decode(tokens)
                return output, None, "eot_err"
            try:
                decoded_token = enc.decode([xcol.item()])
                if "}" in decoded_token: 
                    break
            except KeyError as e:
                print(f"Decoding failed for token ID: {xcol.item()} with error {str(e)}")
                return None, None, "decode_err"
    else:
        tokens = xgen[0, :max_length].tolist()
        output = enc.decode(tokens)
        return output, None, "length_err"
    tokens = xgen[0, :max_length].tolist()
    output = enc.decode(tokens)
    matches = re.findall(r'\{(.*?)\}', output)
    if not matches:
        return output, None, "bracket_err"
    output = "{" + matches[0] + "}"
    try:
        solution = json.loads(output).get("Moves")
        if solution:
            return output, solution, None
        else:
            return output, None, "moves_err"
    except:
        return output, None, "json_err"


def do_puzzle_bench(device, device_type, enc, model, ddp_rank):
    results = {"elo": 100, "results": [], "score": {"total": 0, "score": 0,"avg": 0}, "phase": {"total": 0, "score": 0, "avg": 0}, "openingtags": {"total": 0, "score": 0, "avg": 0}, "goal": {"total": 0, "score": 0, "avg": 0}, "motif": {"total": 0, "score": 0, "avg": 0}, "length": {"total": 0, "score": 0, "avg": 0}, "mate": {"total": 0, "score": 0, "avg": 0}, "moves": {"total": 0, "score": 0, "avg": 0}, "rating": {"total": 0, "total_diff": 0, "avg": 0}}

    puzzles = pd.read_csv(f"Puzzles/Benchmark/puzzle_bench.csv")

    for index, puzzle in puzzles.iterrows():
        results["score"]["total"] += 1

        output, generated_moves, err = do_puzzle(puzzle["FEN"], device, device_type, enc, model, ddp_rank)
        
        if err:
            score = 0
            output_dict = {}
            results["elo"] = max(calculate_elo(results["elo"], puzzle["Rating"], 0), 100)
        else:
            output_dict = json.loads(output)
            score = evaluate_puzzle(generated_moves, puzzle["Moves"])
            results["score"]["score"] += score
            results["elo"] = max(calculate_elo(results["elo"], puzzle["Rating"], score), 100)

        results["results"].append(score)
        results["moves"]["total"] += len(puzzle["Moves"].split())
        results["moves"]["score"] += len(puzzle["Moves"].split()) * score

        results["phase"]["total"] += 1 if puzzle.get("Phase") else 0
        results["phase"]["score"] += 1 if (puzzle.get("Phase") and output_dict.get("Phase") == puzzle.get("Phase")) else 0

        if puzzle.get("OpeningTags"):
            for openingtag in str(puzzle.get("OpeningTags")).split():
                results["openingtags"]["total"] += 1 
                results["openingtags"]["score"] += 1 if (output_dict.get("OpeningTags") and openingtag in output_dict.get("OpeningTags")) else 0

        results["goal"]["total"] += 1 if puzzle.get("Goal") else 0
        results["goal"]["score"] += 1 if (puzzle.get("Goal") and output_dict.get("Goal") == puzzle.get("Goal")) else 0

        if puzzle.get("Motif"):
            for motif in str(puzzle.get("Motif")).split():
                results["motif"]["total"] += 1 
                results["motif"]["score"] += 1 if (output_dict.get("Motif") and motif in output_dict.get("Motif")) else 0

        results["length"]["total"] += 1 if puzzle.get("Length") else 0
        results["length"]["score"] += 1 if (puzzle.get("Length") and output_dict.get("Length") == puzzle.get("Length")) else 0

        if puzzle.get("Mate"):
            for mate in str(puzzle.get("Mate")).split():
                results["mate"]["total"] += 1 
                results["mate"]["score"] += 1 if (output_dict.get("Mate") and mate in output_dict.get("Mate")) else 0

        if puzzle.get("Rating") and output_dict.get("Rating"):
            results["rating"]["total"] += 1 
            results["rating"]["total_diff"] += abs(int(puzzle.get("Rating")) - int(output_dict.get("Rating"))) 

    if results["score"]["total"]:
        results["score"]["avg"] = results["score"]["score"] / results["score"]["total"]
    if results["phase"]["total"]:
        results["phase"]["avg"] = results["phase"]["score"] / results["phase"]["total"]
    if results["openingtags"]["total"]:
        results["openingtags"]["avg"] = results["openingtags"]["score"] / results["openingtags"]["total"]
    if results["goal"]["total"]:
        results["goal"]["avg"] = results["goal"]["score"] / results["goal"]["total"]
    if results["motif"]["total"]:
        results["motif"]["avg"] = results["motif"]["score"] / results["motif"]["total"]
    if results["length"]["total"]:
        results["length"]["avg"] = results["length"]["score"] / results["length"]["total"]
    if results["mate"]["total"]:
        results["mate"]["avg"] = results["mate"]["score"] / results["mate"]["total"]
    if results["moves"]["total"]:
        results["moves"]["avg"] = results["moves"]["score"] / results["moves"]["total"]
    if results["rating"]["total"]:
        results["rating"]["avg"] = results["rating"]["total_diff"] / results["rating"]["total"]

    return results


def do_single_puzzle_bench(elo, device, device_type, enc, model, ddp_rank):
    puzzles = pd.read_csv(f"Puzzles/Benchmark/puzzle_bench.csv")

    # Select a single random puzzle
    random_puzzle = puzzles.sample(n=1).iloc[0]

    output, generated_moves, err = do_puzzle(random_puzzle["FEN"], device, device_type, enc, model, ddp_rank)
    if err:
        score = 0
        new_elo = max(calculate_elo(elo, random_puzzle["Rating"], 0), 100)
    else:
        score = evaluate_puzzle(generated_moves, random_puzzle["Moves"])
        new_elo = max(calculate_elo(elo, random_puzzle["Rating"], score), 100)
    
    puzzle_str = '{' + ','.join(f'"{key}":"{value}"' for key, value in random_puzzle.items() if pd.notna(value)) + '}'
    
    print(f"\nscore: {score} | elo: {new_elo}\npuzzle: {puzzle_str}\noutput: {output}\n")
    bench_log_file = "Log/bench_log.csv"
    with open(bench_log_file, "a", newline='', encoding='utf-8', errors='ignore') as f:
        writer = csv.writer(f)
        
        # If the file is empty, write the header
        if os.path.getsize(bench_log_file) == 0:
            writer.writerow(["Time", "Score", "ELO", "Error", "Puzzle", "Output"])
        
        # Write the data as a row in the CSV file
        writer.writerow([
            time.strftime('%x %X', time.gmtime()),  # Time
            score,                                  # Score
            new_elo,                                # ELO
            err,                                    # Errortype
            puzzle_str,                             # Full Puzzle
            output                                  # Generated Output
        ])    
    return score, new_elo



    
if __name__ == "__main__":
    from Tokenizer import Tokenizer 
    from Model import GPTConfig, GPT

    model_paths = ["Log/ckpt_2500.pt","Log/ckpt_5000.pt","Log/ckpt_7500.pt","Log/ckpt_10000.pt","Log/ckpt_12500.pt","Log/ckpt_15000.pt","Log/ckpt_17500.pt","Log/ckpt_22509.pt"]
    device = 'cuda'
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    enc = Tokenizer()
    ddp_rank = 0

    block_size = 1024
    vocab_size = 1152
    n_layer = 16
    n_head = 16
    n_embd = 768
    dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias = False # do we use bias inside LayerNorm and Linear layers?

    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout)
    for model_path in model_paths:
        checkpoint = torch.load(model_path, map_location=device)
        checkpoint_model_args = checkpoint.get('model_args')
        if not checkpoint_model_args:
            checkpoint_model_args = model_args
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        results = do_puzzle_bench(device, device_type, enc, model, ddp_rank)
        print(f"\n{model_path} -> \n{results}\n")
