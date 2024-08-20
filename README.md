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
To measure the performance of the models, I created a validation dataset of 1000 puzzles (not included in the training data) to evaluate the output when the model is only given the FEN. Ideally, I should have properly evaluated the models after exactly one epoch (5627 iterations) to ensure a fair comparison, but I forgot to save the model at that point. The closest checkpoint I have is at 5k iters.

| **Metric**                 | **Shuffled 2.5k** | **Staged 2.5k** | **Shuffled 5k** | **Staged 5k** | **Shuffled Final**| **Random Guessing** |
|----------------------------|-------------------|-----------------|-----------------|---------------|-------------------|---------------------|
| **ELO**                    | 1010              | 1252            | 1475            | 934           | 1827              | N/A                 | 
| **Score**                  | 8.8%              | 12.0%           | 20.8%           | 6.4%          | 39.0%             | N/A                 |
| **Phase Identification**   | 96.6%             | 96.0%           | 97.5%           | 97.4%         | 99.3%             | 11.11%              |
| **Opening Identification** | 0.0%              | 0.0%            | 0.0%            | 6.6%          | 8.2%              | N/A                 |
| **Goal Identification**    | 38.9%             | 38.3%           | 47.0%           | 38.8%         | 58.0%             | 25%                 |
| **Motif Identification**   | 14.1%             | 13.3%           | 19.7%           | 10.9%         | 28.0%             | 4.35%               |
| **Length Identification**  | 47.9%             | 53.5%           | 53.6%           | 46.6%         | 58.0%             | 25%                 |
| **Mate Pattern Recognition**| 8.5%             | 6.1%            | 11.7%           | 10.6%         | 19.3%             | 7.69%               |
| **Move Sequence Prediction**| 6.4%             | 9.5%            | 16.2%           | 5.1%          | 31.3%             | N/A                 |
| **Puzzle Rating Estimation**| ±608             | ±454            | ±518            | ±683          | ±453              | ±650                |

From this table you can see by the ELO and Score that the staged approach was indeed faster at "converging/learning" due to training on the easier puzzles first. However, when it was later trained on harder puzzles, its performance dramatically decreased. In contrast, the performance of the shuffled approach steadily improved.


## Conclusion
I do not think much can be concluded from this experiment as this is one very specific task and I only did two training runs (dont really have the money to experiment further). But at least in this case its clearly more advantageous to have data shuffled for better results and to prevent overfitting on a subset of the dataset. Although I still believe that curriculum learning might offer some advantages in some cases, incorporating some degree of shuffling seems essential.

## TODO 
- **Experiment with Different Amounts of Difficulty Stages:** Instead of dividing the dataset into exactly 10 difficulty levels, experiment with different numbers of subsets (e.g., 2, 3, 5,... stages) to see if this influences the effectiveness of curriculum learning.
- **Hybrid Curriculum Learning:** Implement a curriculum learning approach where each stage includes a certain percentage of puzzles from other difficulty levels. This could help in preventing overfitting while still guiding the model through a structured learning process.
- **Implement Regularization Techniques:** Add dropout or other regularization techniques during training to mitigate overfitting, especially in the staged approach.
- ...


## Credit
Most of this code is borrowed from [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) and [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT).

