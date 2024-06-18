# Forknight
Forknight aims to create a highly specialized chess dataset leveraging various data types to maximize utility.

## Why
Many people argue that transformers are limited by the exponential amount of data they require and that we've already exhausted a significant portion of the available data. To explore whether this is indeed a serious bottleneck, I wanted to investigate ways to maximize the utility of the data we have and how to generate more. Chess seemed like an excellent starting point for several reasons:
 - The domain is well-defined and narrow.
 - High-volume datasets are readily available online.
 - High-quality data can be sourced from chess engines.
 - Simple evaluation functions make self-play and synthetic data generation straightforward.
 - I would love to be able to use an expert-level chess tutor that is cheap, fast, and superior to any human.


## Dataset
This project contains the code used to create the datasets and some examples.
You can find the full datasets here [to do].

### Data Structure
**Instruction:** Specifies the task to be performed.
**Input:** Provides the necessary context or initial data.
**Output:** Contains the expected result based on the instruction and input.

For about half of the examples, the instruction begins with "Return," where the output is strictly the requested information. Without "Return," the outputs may include additional text, simulating interaction with an assistant.


## Tasks

 - **Find best move** the model is asked to give the best move in a position. This task should probably increase the models intuition, ability to calculate and reasoning.
 - **Give evaluation** the model is asked to evaluate the position. This task should probably increase the models intuition, ability to calculate and reasoning.
 - **Find computerlines** the model is asked to give the top n computerlines with their corresponding evaluation. This task should probably increase the models intuition, ability to calculate and reasoning.
 - **Annotate position** the model is asked to annotate a position.
 - **Guess ELO:** the model is asked to guess the elo of the players based on a string of moves. This task could be helpful for estimating a players elo based on a very low number of games.-
 - **Guess player** the model is asked to guess the names of the players based on a sequence of moves (masters/engines).
 - **Guess moves from position:** the model is asked to predict the following sequence of n move(s) based on the elo rating of the players (can also be names of players for masters/engines). This task could be useful for an enine at specific ELO as it would probably play much more natural and human-like at specific elos.
 - **Guess moves between positions** the model is asked to guess the sequence of moves that connects two positions based on the elo rating of the players. This task should be useful for training the models ability to calculate and plan ahead.
 - **Solve puzzle** the model is asked to solve a chess puzzle (estimate the rating, puzzlethemes, openingtags). This is probably the most effective way to train the model's calculation skills and intuition.
 - **Generate puzzle** the model is asked to generate a chess puzzle.

## Data
In this project I made distinctions between a few types of data: raw data, transformed data, augmented data and virtual data (these last two are also called synthetic data).

### Raw Data
The kind of data you can find on and scrape from the web. This kind of data has proven very useful for example during pretraining of large language models. 
I do however think this pretraining can be a bit wasteful if you want a model that is more like an assistant and not just a compression of the internet.

### Transformed Data
With transformed data I mean data that is based upon raw data but where some simple techniques are used to turn it into some task that you want the assistant model to be able to do.
This is the kind of data that is most often used for finetuning pretrained LLM's.

### Augmented Data
Data that is based upon either raw or transformed data but where an LLM is used to annotate or label it. To create this data you can explain to the LLM what it should do plus some examples and then you feed it the raw/transformed data it should base its annotations on. 

### Virtual Data
This kind of data is not based upon existing data, it is based upon the imagination of an LLM (I know the imagination of the LLM is based on existing data but you get what I mean). The difference between augmented and virtual data is that for virtual data you don't give the model data to base its annotations on, only an explanation plus examples.




