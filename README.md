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
You can probably also see in this graph that towards the end of the second epoch the model suddenly went completely of the rails to the point where it would just spam random tokens, this is probably due to overfitting on the easier puzzles and this messes up the optimisation process when it arrives at the harder puzzles.

![image](https://github.com/user-attachments/assets/ae0deb1e-103d-4800-89ec-94ee21886b49)

In the second training run I completely shuffled the puzzles and trained for 4 epochs.

![image](https://github.com/user-attachments/assets/6242f9e1-50fd-4440-853d-481fe8463b0b)


For comparison:

![image](https://github.com/user-attachments/assets/cb955be7-e782-454e-9956-3fdb30f02108)

![image](https://github.com/user-attachments/assets/39c1b3ec-9163-4d5a-bce7-0575040fd7ff)


## Benchmark
I evaluated the final model from the second training run using a set of 1000 test puzzles, measuring its performance on several key metrics:
1. **Move Sequence Prediction:** The model correctly predicted 1152 out of 3686 moves, achieving a score of approximately 390/1000 (partial credit for partially correct sequences). This corresponds to a puzzle rating (Elo) of around 1826.
2. **Phase Identification:** The model was pretty good at recognizing the game phase (e.g., opening, middlegame, rook endgame, etc.), with an accuracy of 99.3%.
3. **Opening Identification:** The model struggled with identifying specific chess openings, achieving a correct identification rate of only 8.2%.
4. **Goal and Length Prediction:** The model correctly identified the puzzle’s goal (e.g., mate, crushing, advantage, equality) and the expected move sequence length (e.g., one-move, short, long, very long) in about 58% of the puzzles.
5. **Motif Identification:** The model correctly identified tactical motifs (e.g., sacrifice, advanced pawn, fork, etc.) in 28% (345/1230) of cases.
6. **Mate Pattern Recognition:** The model correctly identified mate patterns (e.g., back-rank mate, Boden's mate, mate in 3, etc.) in 19% (201/1040) of cases.
7. **Puzzle Rating Estimation:** On average, the model's estimate of the puzzle rating was off by 453 Elo points.

## Conclusion
I do not think much can be concluded from this experiment as I only did two runs (dont have the money to experiment further). But at least in this case it seems more advantageous to have data shuffled to prevent overfitting. Although I still believe that curriculum learning might offer some advantages in some cases, incorporating some degree of shuffling seems essential to avoid model collapse and overfitting.


## Credit
Most of this code is borrowed from [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) and [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT).

