# Lab 06: Training

We have data, a tokenizer, an empty model, and the math to compute gradients.
Now we train.

Training means showing the model sentences and adjusting its parameters to make
better predictions. Each training step works like this:

## Step by Step

### 1. Pick a Sentence and Tokenize It

```typescript
const doc = docs[step % docs.length];   // e.g., "the cat eats a muffin"
const tokens = tokenizer.encode(doc);    // [596, 541, 89, 152, 0, 358, 596]
```

### 2. Feed Tokens One at a Time and Predict the Next One

For each position, we give the model the current token and ask it to predict
the next:

```
Position 0: input = BOS(596)     -> model predicts -> target = "the"(541)
Position 1: input = "the"(541)   -> model predicts -> target = "cat"(89)
Position 2: input = "cat"(89)    -> model predicts -> target = "eats"(152)
Position 3: input = "eats"(152)  -> model predicts -> target = "a"(0)
Position 4: input = "a"(0)      -> model predicts -> target = "muffin"(358)
Position 5: input = "muffin"(358)-> model predicts -> target = BOS(596)
```

At each position, the model outputs 597 logits, one score per word. We want
the score for the *correct* next word to be high.

### 3. Compute the Loss

The loss measures how wrong the model is. We use **cross-entropy loss**: for
each position, we convert the logits to probabilities (via softmax), then take
the negative log of the probability assigned to the correct next token.

Why negative log? Because it has exactly the shape we want: when the model is
perfectly confident in the right answer (probability 1.0), the loss is
`-log(1) = 0`. As confidence drops, the loss grows, slowly at first, then
faster and faster toward infinity. A model that assigns 50% to the right word
gets a mild penalty, but a model that assigns 1% gets hammered. This is what
pushes the model to avoid being confidently wrong.

```typescript
const logits = gpt(model, tokenId, posId, keys, values);
const probs = softmax(logits);
losses.push(probs[targetId].log().neg());
```

If the model assigns probability 0.9 to the correct word, the loss is
`-log(0.9) = 0.105` (low, good). If it assigns probability 0.01, the loss
is `-log(0.01) = 4.605` (high, bad).

The total loss for the sentence is the average across all positions:

```typescript
const loss = vsum(losses).div(n);
```

### 4. Compute Gradients (Backward Pass)

One call to `backward()` fills in the gradient for every parameter:

```typescript
loss.backward();
```

After this, each of the 63,296 parameters knows: "if I increase, the loss goes
up/down by this much."

### 5. Update Parameters (Adam Optimizer)

The simplest approach would be `param -= learningRate * grad`, just moving each
parameter a small step in the direction the gradient points. But gradients are
noisy: each step only sees one sentence, so the gradient for any single step
might point in a slightly wrong direction. We want to smooth out that noise
and also adapt the step size for each parameter individually. Parameters with
consistently large gradients should take smaller, more careful steps.

That is what the **Adam optimizer** does:

```typescript
for (let i = 0; i < params.length; i++) {
  mBuf[i] = beta1 * mBuf[i] + (1 - beta1) * params[i].grad;
  vBuf[i] = beta2 * vBuf[i] + (1 - beta2) * params[i].grad ** 2;
  const mHat = mBuf[i] / (1 - beta1 ** (step + 1));
  const vHat = vBuf[i] / (1 - beta2 ** (step + 1));
  params[i].data -= lrT * mHat / (Math.sqrt(vHat) + epsAdam);
  params[i].grad = 0;  // reset for next step
}
```

It maintains two running averages for each parameter:

- `mBuf`: the average gradient direction (momentum: keeps moving even if the
  current gradient is noisy)
- `vBuf`: the average gradient magnitude (adapts the step size per parameter,
  so parameters with large gradients take smaller steps)

The learning rate also decays linearly from 0.01 to 0 over the course of
training, so the model takes large steps early (to learn quickly) and small
steps later (to fine-tune).

## Watching It Learn

When training runs, the loss steadily decreases:

```
step    1 / 5000 | loss 6.3917
step  500 / 5000 | loss 3.2184
step 1000 / 5000 | loss 2.8549
step 2000 / 5000 | loss 2.4012
step 3000 / 5000 | loss 2.1538
step 4000 / 5000 | loss 1.9107
step 5000 / 5000 | loss 1.7623
```

The initial loss of ~6.39 corresponds to random guessing: `-log(1/597) = 6.39`.
The model has no idea which of the 597 words comes next, so it assigns roughly
equal probability to all of them. As training progresses, the model learns to
assign higher probability to the correct word, and the loss drops.

A loss of ~1.76 means the model is, on average, assigning about `e^(-1.76) =
17%` probability to the correct next word. Not perfect, but far better than
the 0.17% it started with, a 100x improvement.

## The Training Configuration

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
  docs,
  tokenizer,
);
```

5,000 steps takes a few minutes on a laptop. Each step processes one sentence,
building a full computation graph from tokens to loss, running backpropagation,
and updating all parameters.

Next: [Lab 07: Saving the Model](07-saving.md)
