# The Generation Loop

Generation works by starting with the BOS token and repeatedly asking the
model: "given everything so far, what word comes next?"

```typescript
let tokenId = tokenizer.BOS;      // start with BOS
const tokens: number[] = [];

for (let posId = 0; posId < model.config.blockSize; posId++) {
  const logits = gpt(model, tokenId, posId, keys, values);

  // 1. Temperature scaling
  let scores = logits.map((l) => l.data / temperature);

  // 2. Top-k: keep only the k highest scores
  if (topK > 0 && topK < scores.length) {
    const sorted = [...scores].sort((a, b) => b - a);
    const cutoff = sorted[topK];
    scores = scores.map((s) => (s >= cutoff ? s : -Infinity));
  }

  // 3. Top-p: keep smallest set whose probabilities sum to p
  if (topP < 1.0) { /* accumulate sorted probs, zero out the rest */ }

  // 4. Softmax and sample (plain math, no autograd needed at inference)
  const maxS = Math.max(...scores.filter((s) => s !== -Infinity));
  const exps = scores.map((s) => (s === -Infinity ? 0 : Math.exp(s - maxS)));
  const total = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => e / total);
  tokenId = weightedChoice(probs);
  if (tokenId === tokenizer.BOS) break;
  tokens.push(tokenId);
}

return tokenizer.decode(tokens);
```

The full pipeline for turning logits into a sampled token:

```
597 logits -> temperature scaling -> top-k filter -> top-p filter -> softmax -> weighted sample
```

## Step by Step

**1. Forward pass.** Feed the current token and position into the model. It
returns 597 logits, one raw score per word in the vocabulary.

**2. Temperature scaling.** Divide all logits by the temperature. This
controls how "creative" the model is:

| Temperature | Effect |
|---|---|
| 0.3 (low) | Sharper probabilities. Model picks the most likely word almost every time. Repetitive but safe. |
| 0.8 (medium) | Balanced. Some variety, mostly coherent. |
| 1.5 (high) | Flatter probabilities. Model considers unlikely words. More creative but may produce nonsense. |

**3. Top-k filter.** If `topK` is set (e.g., 10), sort the scores and set
everything outside the top *k* to negative infinity. Those words get zero
probability after softmax and can never be chosen.

```
597 scores -> keep top 10 -> set 587 to -infinity
```

This hard cutoff prevents the model from ever picking a very unlikely word.
The downside is that a fixed *k* does not adapt. Sometimes the model is very
confident and only 2-3 words make sense; other times, 50 words are plausible.

**4. Top-p filter (nucleus sampling).** If `topP` is set (e.g., 0.9),
compute probabilities from the current scores, sort by probability, and
accumulate until the sum reaches *p*. Zero out everything past the cutoff.

```
scores -> softmax -> sort -> accumulate until sum >= 0.9 -> zero out the rest
```

If the model is confident ("the" has probability 0.85), maybe only 2-3 words
survive. If the model is uncertain and the distribution is flat, maybe 50
words survive. Top-p adapts to the model's confidence at each position, which
is why it is generally preferred over top-k.

**5. Softmax and sample.** Convert the filtered scores to a probability
distribution and randomly pick a word weighted by those probabilities. The word
"the" might have probability 0.3 and "a" might have 0.15. The model does not
always pick the top choice, which gives the output variety.

**6. Check for BOS.** If the sampled token is BOS, the sentence is done.
Otherwise, feed this new token back in and repeat.

**7. Decode.** Convert the collected token IDs back to words.
