# Softmax (Scores to Probabilities)

Neural networks often produce raw, unnormalized scores: numbers that can be
positive, negative, large, or small. On their own these scores don't mean
much. Softmax converts a vector of scores into a **probability distribution**:
a vector of positive numbers that sum to 1.

```typescript
function softmax(logits: Value[]): Value[] {
  const maxVal = Math.max(...logits.map((v) => v.data));
  const exps = logits.map((v) => v.sub(maxVal).exp());
  const total = vsum(exps);
  return exps.map((e) => e.div(total));
}
```

The steps: subtract the maximum value (for numerical stability), exponentiate
each element, then divide by the sum. For example:

```
scores = [2.0, 1.0, 0.1]

subtract max:  [0.0, -1.0, -1.9]
exponentiate:  [1.0,  0.37, 0.15]
divide by sum: [0.66, 0.24, 0.10]
```

The largest score gets the highest probability, but every element stays
positive and they all sum to 1.

Softmax is used in three places:
1. Inside attention, to turn attention scores into attention weights
2. During training, to turn output scores into probabilities for computing the loss
3. During generation, to turn output scores into a probability distribution for sampling
