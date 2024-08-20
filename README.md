# Forknight
This project was created to experiment with the effects of 'curriculum learning' as opposed to random sampling for pretraining LLM's.
My hypothesis was that a model would learn faster/more computeefficient when starting of the training on easier samples and working its way up to the harder samples.
To test this hypothesis I created a small model (114M params) and trained it on 3.8M Lichess puzzles in the following format:
> {"FEN":"r1bqk1nr/ppp2ppp/3p2n1/3Pp3/1bP1P3/2N3P1/PP3P1P/R1BQKBNR w KQkq - 0 7",
> "Phase":"opening","OpeningTags":"Nimzowitsch Defense Kennedy Variation",
> "Goal":"crushing","Motif":"fork","Length":"short","Moves":"d1a4 c8d7 a4b4","Rating":"1126"}

## Experiment
For the first training run I divided the puzzles into 10 different stages based on their rating and trained the model for 2 epochs.
If you look closely at this graph you can somewhat see those stages in the loss values.
You can probably also see in this graph that towards the end of the second epoch the model suddenly went completely of the rails to the point where it would just spam random tokens, this is probably due to overfitting on the easier puzzles (puzzles of similar rating probably have very similar themes) and this messes up the optimisation process when it arrives at the harder puzzles.

The better approach would probably be to train for a single epoch (or less) with increasing difficulty and from that point on train on a fully shuffled dataset until an optimal validation loss is achieved. Or have some amount of puzzles from other stages sprinkled into the mix of each stage.

![image](https://github.com/user-attachments/assets/ae0deb1e-103d-4800-89ec-94ee21886b49)

In the second training run I completely shuffled the puzzles and trained for 4 epochs. As you can see this graph is much smoother than the first one.

![image](https://github.com/user-attachments/assets/6242f9e1-50fd-4440-853d-481fe8463b0b)

For comparison:

![image](https://github.com/user-attachments/assets/cb955be7-e782-454e-9956-3fdb30f02108)

![image](https://github.com/user-attachments/assets/39c1b3ec-9163-4d5a-bce7-0575040fd7ff)



## Benchmark
I definitely should have properly evaluated the models after exactly one epoch (5627 iters) to have a fair comparison, but I forgot to save the model at that point. The closest checkpoint I have is at 5k iters:
| Metric | Staged (5k)     | Shuffled (5k)     |
|------------|---------------------|-----------------------|
| **ELO**    | 934                 | 1475                  |
| **SCORE**  | 6.4%                | 20.8%                 |
| **PHASE**  | 97.4%               | 97.5%                 |
| **OPENING**| 6.6%                | 0.0%                  |
| **GOAL**   | 38.8%               | 47.0%                 |
| **MOTIF**  | 10.9%               | 19.7%                 |
| **LENGTH** | 46.6%               | 53.6%                 |
| **MATE**   | 10.6%               | 11.7%                 |
| **MOVES**  | 5.1%                | 16.2%                 |
| **RATING** | 683                 | 518                   |


I also evaluated the final model from the second training run using a set of 1000 test puzzles by giving it only the FEN's, to measure its performance:
1. **Move Sequence Prediction:** The model correctly predicted 1152 out of 3686 moves, achieving a score of approximately 390/1000 (partial credit for partially correct sequences). This corresponds to a puzzle rating (Elo) of around 1826*. (*not very accurate but was too lazy to correctly implement it)
2. **Phase Identification:** The model was pretty good at recognizing the game phase (e.g., opening, middlegame, rook endgame, etc.), with an accuracy of 99.3%. (random uniform guessing would be 11.11%)
3. **Opening Identification:** The model struggled with identifying specific chess openings, achieving a correct identification rate of only 8.2%. 
4. **Goal Identification:** The model correctly identified the puzzle’s goal (e.g., mate, crushing, advantage, equality) in about 58% of the puzzles. (random uniform guessing would be 25%)
5. **Length Identification:** The model correctly identified both the puzzle’s move sequence length (e.g., one-move, short, long, very long) in about 58% of the puzzles. (random uniform guessing would be 25%)
6. **Motif Identification:** The model correctly identified tactical motifs (e.g., sacrifice, advanced pawn, fork, etc.) in 28% (345/1230) of cases. (random uniform guessing would be 4.35%)
7. **Mate Pattern Recognition:** The model correctly identified mate patterns (e.g., back-rank mate, Boden's mate, mate in 3, etc.) in 19% (201/1040) of cases. (random uniform guessing would be 7.69%)
8. **Puzzle Rating Estimation:** On average, the model's estimate of the puzzle rating was off by 453 Elo points.

## Conclusion
I do not think much can be concluded from this experiment as this is one very specific task and I only did two training runs (dont really have the money to experiment further). But at least in this case its clearly more advantageous to have data shuffled for better results and to prevent overfitting on a subset of the dataset. Although I still believe that curriculum learning might offer some advantages in some cases, incorporating some degree of shuffling seems essential.

## TODO 
- **Experiment with Different Amounts of Difficulty Stages:** Instead of dividing the dataset into exactly 10 difficulty levels, experiment with different numbers of subsets (e.g., 2, 3, or 5 stages) to see if this influences the effectiveness of curriculum learning.
- **Hybrid Curriculum Learning:** Implement a curriculum learning approach where each stage includes a certain percentage of puzzles from other difficulty levels. This could help in preventing overfitting while still guiding the model through a structured learning process.
- **Implement Regularization Techniques:** Add dropout or other regularization techniques during training to mitigate overfitting, especially in the staged approach.
- ...


## Credit
Most of this code is borrowed from [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) and [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT).

