# Lab 08: Generation

The model is trained. The weights are saved. Now we load them and generate new
sentences.

## The Generation Loop

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

  // 4. Softmax and sample
  const probs = softmax(scores);
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

## Comparing Sampling Strategies

| Strategy | What it does | Trade-off |
|---|---|---|
| Temperature | Reshapes the entire distribution | Simple, but low-probability junk words can still be picked |
| Top-k | Hard cutoff at *k* words | Fixed *k* does not adapt to model confidence |
| Top-p | Adaptive cutoff by cumulative probability | Handles both confident and uncertain positions well |

In practice, these are combined. A typical production configuration might use
`temperature=0.7`, `top_p=0.9`, and `top_k=50` together. Temperature reshapes
the distribution, then top-k and top-p trim the tail. Our implementation
supports all three, individually or combined.

## The KV Cache

You might notice that generation calls `gpt()` once per token in the sentence.
Each call processes only the *current* token, not the whole sentence. So how
does the model know about previous words?

The answer is the **KV cache**: key and value vectors from the attention
mechanism that are saved and reused:

```typescript
const { keys, values } = createKVCache(model);
```

Each time we process a token, the model computes key (K) and value (V) vectors
for the attention layers and appends them to the cache. When computing
attention for the next token, the model looks at all cached K/V pairs from
previous positions. This means:

- Position 0 (BOS): no context, just the starting signal
- Position 1 ("the"): attention can look back at BOS
- Position 2 ("cat"): attention can look back at BOS and "the"
- Position 3 ("eats"): attention can look back at BOS, "the", and "cat"

The KV cache is what makes token-by-token generation efficient. Without it, we
would have to reprocess the entire sentence from scratch at every position.

## Example Output

With temperature 0.8, the trained model produces sentences like:

```
 1. the brown horse sits
 2. we fly to the store
 3. the girl reads a card
 4. mom cooked the muffin
 5. the kitten runs fast
 6. seven ducks swim at dusk
 7. i like the tall tree
 8. the cow eats
```

These are not sentences from the training data. The model invented them. But
they follow the same patterns: articles before nouns, verbs in the right place,
reasonable word combinations. The model learned the structure of simple English
from 30,000 examples and 5,000 training steps.

Next: [Lab 09: Smoke Test](09-smoke-test.md)
