# Lab 04: Neural Network Primitives

On top of the autograd engine from Lab 03, we build three operations that the
model uses throughout. These are general-purpose building blocks, the same
operations found in every neural network framework. A neural network only needs
a surprisingly small toolkit:

1. **A way to combine inputs.** Take several numbers, multiply each by a
   learned weight, and add the results together. This is just a weighted sum,
   the same idea as computing a course grade from weighted assignment scores.
   That is what `linear` does.

2. **A way to make decisions.** The model needs to express confidence: "I
   think the next word is 70% likely to be *cat* and 20% likely to be *dog*."
   Raw scores can be any number, but probabilities must be positive and sum to
   1. That is what `softmax` does. It turns arbitrary scores into a proper
   probability distribution.

3. **A way to stay stable.** Numbers that pass through dozens of weighted sums
   tend to either explode toward infinity or shrink toward zero. That is what
   `rmsnorm` does. It rescales a vector so the values stay in a reasonable
   range, similar to normalizing a set of exam scores to have a consistent
   spread.

## Linear (Matrix-Vector Multiply)

The fundamental neural net operation. It transforms a vector by multiplying it
with a weight matrix:

```typescript
function linear(x: Value[], w: Matrix): Value[] {
  return w.map((wo) => vsum(wo.map((wi, i) => wi.mul(x[i]))));
}
```

Each output element is a dot product, a weighted sum of the inputs. This is
how the model mixes information across dimensions. A weight matrix with shape
128 x 32 takes a 32-element input vector and produces a 128-element output
vector.

Because every multiplication and addition uses `Value` nodes, the entire
operation is differentiable, so gradients flow back through it during training.

## Softmax (Scores to Probabilities)

The model produces raw, unnormalized scores called **logits**, one per token
in the vocabulary. If the vocabulary has 597 tokens, the model outputs 597
logits, each representing how likely that token is to come next. A logit can be
any number (positive, negative, large, small) and on its own it doesn't
mean much. Softmax converts a vector of logits into a probability distribution
that sums to 1:

```typescript
function softmax(logits: Value[]): Value[] {
  const maxVal = Math.max(...logits.map((v) => v.data));
  const exps = logits.map((v) => v.sub(maxVal).exp());
  const total = vsum(exps);
  return exps.map((e) => e.div(total));
}
```

The steps: subtract the maximum value (for numerical stability), exponentiate
each element, then divide by the sum. The result is a vector of positive
numbers that sum to 1, a probability distribution.

Softmax is used in three places:
1. Inside attention, to turn attention scores into attention weights
2. During training, to turn output scores into probabilities for computing the loss
3. During generation, to turn output scores into a probability distribution for sampling

## RMSNorm (Normalization)

Rescales a vector to have roughly unit variance. This prevents numbers from
growing too large or too small as they pass through multiple layers:

```typescript
function rmsnorm(x: Value[]): Value[] {
  const ms = vsum(x.map((xi) => xi.mul(xi))).div(x.length);
  const scale = ms.add(1e-5).pow(-0.5);
  return x.map((xi) => xi.mul(scale));
}
```

It computes the root mean square of the vector, then divides each element by
it. The `1e-5` prevents division by zero. Without normalization, activations
can explode or vanish across layers, making training unstable.

## Summary

These three primitives (`linear`, `softmax`, and `rmsnorm`) are the
building blocks the model assembles in Lab 05. Each one is built entirely from
`Value` operations, so the autograd engine can compute gradients through all
of them.

| Primitive | What it does | Where it is used |
|---|---|---|
| `linear` | Matrix-vector multiply | Attention projections, MLP layers, output head |
| `softmax` | Scores to probabilities | Attention weights, training loss, generation sampling |
| `rmsnorm` | Normalize to unit variance | Before each transformer sub-block |

Next: [Lab 05: The Model](05-model.md)
