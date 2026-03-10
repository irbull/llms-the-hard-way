# The Training Loop

Each iteration of the training loop processes one sentence through five stages:
pick a sentence, feed its tokens through the model, measure the error, compute
gradients, and update the weights. Let's walk through each part, then see the
complete loop.

## 1. Pick a Sentence and Tokenize It

```typescript
const sentence = sentences[step % sentences.length];
const tokens = tokenizer.encode(sentence);
const n = Math.min(model.config.blockSize, tokens.length - 1);
```

We cycle through the training sentences using modular arithmetic: step 0 picks
sentence 0, step 1 picks sentence 1, and so on, wrapping back to 0 when we
reach the end. The tokenizer turns "the cat eats a muffin" into
`[596, 541, 89, 152, 0, 358, 596]`: a BOS marker, then each word, then BOS
again.

The variable `n` is the number of predictions we will make. We subtract 1 from
the token count because the last token has no "next word" to predict. We also
cap at `blockSize` (16) because the position embedding table only has 16 rows.
The model cannot handle longer sequences.

## 2. Feed Tokens and Predict the Next Word

We create a fresh KV cache for each sentence (each sentence is a standalone
training example; we don't carry context between sentences). Then we process
the sentence token by token. At each position, the model sees all previous
tokens through the cache and outputs 597 scores as its prediction for what
comes next.

Because we have the complete sentence, **we already know the right answer**
at every position. That is what makes training possible: we can measure how
wrong each prediction is.

```
Position 0: feed BOS
  context: [BOS]
  model predicts → 597 scores
  correct answer: "the"

Position 1: feed "the"
  context: [BOS, "the"]
  model predicts → 597 scores
  correct answer: "cat"

Position 2: feed "cat"
  context: [BOS, "the", "cat"]
  model predicts → 597 scores
  correct answer: "eats"

Position 3: feed "eats"
  context: [BOS, "the", "cat", "eats"]
  model predicts → 597 scores
  correct answer: "a"

Position 4: feed "a"
  context: [BOS, "the", "cat", "eats", "a"]
  model predicts → 597 scores
  correct answer: "muffin"

Position 5: feed "muffin"
  context: [BOS, "the", "cat", "eats", "a", "muffin"]
  model predicts → 597 scores
  correct answer: BOS (end of sentence)
```

Each call to `gpt()` builds a computation graph of `Value` nodes: thousands of
multiply, add, and relu operations connected by pointers. All six predictions
within a sentence share the same graph (through the KV cache), so a single
`backward()` call later can propagate gradients through all of them at once.

Early on, the model's scores are essentially random. "The" might rank 400th
out of 597. The loss measures exactly how bad that is.

## 3. Compute the Loss

The loss measures how wrong the model is. We use **cross-entropy loss**: for
each position, we convert the logits to probabilities (via softmax), then take
the negative log of the probability assigned to the correct next token:

```typescript
const probs = softmax(logits);
losses.push(probs[targetId].log().neg());
```

Why negative log? Because it has exactly the shape we want:

- When the model assigns probability **1.0** to the right answer:
  loss = `-log(1) = 0`. Perfect, no penalty.
- When the model assigns probability **0.5**, uncertain but in the ballpark:
  loss = `-log(0.5) = 0.693`. A gentle nudge.
- When the model assigns probability **0.01**, almost entirely wrong:
  loss = `-log(0.01) = 4.605`. A hard shove.

The penalty grows slowly at first, then accelerates toward infinity. This is
what pushes the model to avoid being confidently wrong. A model that puts 90%
probability on the wrong word gets punished far more than one that spreads its
bets.

| Model assigns to correct word | Loss | Meaning |
|---|---|---|
| 0.90 | 0.105 | Low loss: strong, correct prediction |
| 0.50 | 0.693 | Moderate loss: uncertain but reasonable |
| 0.17 | 1.772 | High loss: mostly wrong |
| 0.01 | 4.605 | Very high loss: confidently wrong |

The total loss for the sentence is the average across all positions:

```typescript
const loss = vsum(losses).div(n);
```

Averaging means the model is equally penalized whether the sentence has 3 words
or 15. Otherwise long sentences would dominate the gradient signal and short
sentences would barely contribute to learning.

## 4. Compute Gradients (Backward Pass)

One call to `backward()` fills in the gradient for every parameter:

```typescript
loss.backward();
```

`backward()` returns nothing; it works entirely through side effects. It walks
the computation graph built during the forward pass (all those `Value` nodes
connected by pointers) and sets the `.grad` field on every node that
participated in the computation. After it runs, each parameter's `.grad`
answers the question: *if I increase this parameter by a tiny amount, how much
does the loss change?*

A positive gradient means "increasing this parameter makes the loss worse,"
so the optimizer will decrease it. A negative gradient means "increasing this
parameter makes the loss better," so the optimizer will increase it. A large
gradient means this parameter has a big effect on the current prediction; a
small gradient means it barely matters for this sentence.

### Sparse Embedding Gradients

Not all 63,296 parameters get meaningful gradients from every sentence. The
layer weights (attention, MLP, output projection) are used in full on every
forward pass (`linear()` computes a dot product with every row) so they
always receive gradients. But embedding lookup is just array indexing: only the
rows for tokens that appear in the current sentence enter the computation graph.

For a 7-token sentence like "the cat eats a muffin", only 6 unique token
embedding rows are touched (plus 6 position embedding rows). The remaining 591
token embedding rows keep their `.grad` at 0 and are not updated this step.
Over the full training run, each word appears in many sentences, so every
embedding row eventually gets trained, but frequent words get far more updates
than rare ones.

## 5. Update Parameters (Adam Optimizer)

The simplest approach would be `param -= learningRate * grad`: nudge each
parameter in the direction opposite its gradient. But this has two problems:

1. **Noisy gradients.** Each step only sees one sentence, so the gradient for
   any single step might point in a slightly wrong direction. We want to smooth
   out that noise by remembering which direction gradients have been pointing
   recently.

2. **One size doesn't fit all.** Some parameters have consistently large
   gradients (they are sensitive), others have small ones (they are stable).
   A single learning rate either over-shoots the sensitive ones or under-shoots
   the stable ones.

The **Adam optimizer** solves both problems. It maintains two running averages
for each parameter:

```typescript
mBuf[i] = beta1 * mBuf[i] + (1 - beta1) * params[i].grad;
vBuf[i] = beta2 * vBuf[i] + (1 - beta2) * params[i].grad ** 2;
```

- **`mBuf` (momentum)**: a running average of the gradient direction. With
  `beta1 = 0.85`, each new gradient contributes 15% and the history contributes
  85%. This smooths out noise: if the gradient has been pointing left for
  several steps, the momentum keeps pushing left even if one step says right.

- **`vBuf` (velocity)**: a running average of the squared gradient magnitude.
  This tracks how large the gradients have been for this parameter. Parameters
  with consistently large gradients get a smaller effective step size, and
  parameters with small gradients get a larger one.

### Bias Correction

At the start of training, both `mBuf` and `vBuf` are initialized to zero. That
makes the early running averages severely biased toward zero. After one step,
`mBuf` is only 15% of the first gradient. The bias correction fixes this:

```typescript
const mHat = mBuf[i] / (1 - beta1 ** (step + 1));
const vHat = vBuf[i] / (1 - beta2 ** (step + 1));
```

At step 0: `1 - 0.85^1 = 0.15`, so `mHat` divides by 0.15, effectively
undoing the 85% shrinkage. By step 20, `1 - 0.85^21 ≈ 0.97`, and the
correction is negligible. This ensures the early updates are the right size
from the start.

### The Actual Update

```typescript
params[i].data -= lrT * mHat / (Math.sqrt(vHat) + epsAdam);
params[i].grad = 0;  // reset for next step
```

The update divides the smoothed gradient (`mHat`) by the square root of the
smoothed squared gradient (`vHat`). This ratio normalizes the step: parameters
where gradients are consistently large take proportionally smaller steps.
The `epsAdam` (1e-8) prevents division by zero when `vHat` is tiny.

### Learning Rate Decay

The learning rate also decays linearly from 0.01 to 0 over the course of
training:

```typescript
const lrT = learningRate * (1 - step / numSteps);
```

At step 0, `lrT = 0.01`. At step 2500 (halfway), `lrT = 0.005`. At step 4999,
`lrT ≈ 0`. Large steps early let the model learn quickly. Small steps later
let it fine-tune without overshooting.

Finally, we reset the gradient to zero so the next sentence starts with a clean
slate. Without this, gradients would accumulate across sentences, and each
update would be influenced by all previous sentences rather than just the
current one.

## The Complete Loop

Here is the full training loop with all five stages together:

```typescript
for (let step = 0; step < numSteps; step++) {
  // --- Step 1: Pick a sentence and tokenize ---
  // Cycle through training data; the tokenizer wraps the sentence with BOS markers
  const sentence = sentences[step % sentences.length];
  const tokens = tokenizer.encode(sentence);
  const n = Math.min(model.config.blockSize, tokens.length - 1);

  // --- Step 2: Forward pass ---
  // Feed tokens one at a time, building a computation graph.
  // Each gpt() call adds to the KV cache so later tokens can attend to earlier ones.
  const { keys, values } = createKVCache(model);
  const losses: Value[] = [];

  for (let posId = 0; posId < n; posId++) {
    const tokenId = tokens[posId];
    const targetId = tokens[posId + 1];  // the word we want the model to predict
    const logits = gpt(model, tokenId, posId, keys, values);
    const probs = softmax(logits);
    // Cross-entropy loss: -log(probability assigned to the correct word)
    // High probability on the right word → low loss; low probability → high loss
    losses.push(probs[targetId].log().neg());
  }

  // Average loss over the sentence so short and long sentences contribute equally
  const loss = vsum(losses).div(n);

  // --- Step 3: Backward pass ---
  // Walk the computation graph and fill in .grad on every Value node.
  // After this, each parameter knows how much it contributed to the loss.
  loss.backward();

  // --- Step 4: Adam optimizer with linear learning rate decay ---
  // Large steps early (lr = 0.01) to learn quickly, decaying to 0 to fine-tune
  const lrT = learningRate * (1 - step / numSteps);
  for (let i = 0; i < params.length; i++) {
    // Update momentum (smoothed gradient direction)
    mBuf[i] = beta1 * mBuf[i] + (1 - beta1) * params[i].grad;
    // Update velocity (smoothed gradient magnitude)
    vBuf[i] = beta2 * vBuf[i] + (1 - beta2) * params[i].grad ** 2;
    // Bias correction: counteract the zero-initialization of mBuf/vBuf
    const mHat = mBuf[i] / (1 - beta1 ** (step + 1));
    const vHat = vBuf[i] / (1 - beta2 ** (step + 1));
    // Update: step in the smoothed gradient direction, scaled by the inverse
    // of the gradient magnitude so sensitive parameters take smaller steps
    params[i].data -= lrT * mHat / (Math.sqrt(vHat) + epsAdam);
    params[i].grad = 0;  // reset for next sentence
  }
}
```

Every operation in the forward pass uses `Value` nodes, so the entire path from
token to loss is a single computation graph. When `backward()` runs, gradients
flow back through every step, from the loss, through softmax, through every
layer's attention and MLP blocks, all the way to the embedding lookup. That is
the whole training loop: predict, measure, learn, repeat.
