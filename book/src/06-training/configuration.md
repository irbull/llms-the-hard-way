# The Training Configuration

```typescript
const model = train(
  untrained,
  {
    numSteps: 5000,       // how many sentences to learn from
    learningRate: 0.01,   // how aggressively to update (decays to 0)
    beta1: 0.85,          // momentum decay rate
    beta2: 0.99,          // gradient magnitude decay rate
    epsAdam: 1e-8,        // numerical stability term
  },
  sentences,
  tokenizer,
);
```

5,000 steps takes a few minutes on a laptop. Each step processes one sentence,
building a full computation graph from tokens to loss, running backpropagation,
and updating all parameters.

## What Each Parameter Controls

**`numSteps`** is how many sentences the model trains on. The training loop
cycles through the dataset one sentence per step (`sentences[step %
sentences.length]`). One full pass through every sentence is called an *epoch*.
If you had ~400 unique sentences, 5,000 steps would mean roughly 12 epochs —
cycling through the data 12 times. Our dataset has 30,000 sentences, so 5,000
steps covers only the first sixth with no cycling at all. You can increase
`numSteps` or reduce the dataset size to get more epochs, but be careful:
cycling through the same sentences too many times risks *overfitting*, where the
model memorizes the training data instead of learning general patterns.

**`learningRate`** controls how large each parameter update is. At 0.01, each
step nudges parameters by roughly 1% of the gradient signal. This decays
linearly to 0 over training — large steps early to learn quickly, small steps
later to fine-tune. If the learning rate is too high, the model overshoots and
loss oscillates instead of decreasing. Too low and learning is painfully slow.

**`beta1`** and **`beta2`** control the Adam optimizer's memory. `beta1 = 0.85`
means momentum remembers 85% of recent gradient directions and incorporates 15%
of the new one — this smooths out noisy gradients. `beta2 = 0.99` means the
magnitude tracker is more conservative, updating slowly to get a stable
estimate of each parameter's typical gradient size.

**`epsAdam`** is a tiny constant (0.00000001) added to prevent division by zero
in the Adam update. It never meaningfully affects the training — it is purely a
safety net for numerical stability.
