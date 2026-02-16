# Watching It Learn

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

## What the Numbers Mean

The initial loss of ~6.39 corresponds to random guessing: `-log(1/597) ≈ 6.39`.
The model has no idea which of the 597 words comes next, so it assigns roughly
equal probability to all of them.

As training progresses, the model learns patterns — "the" often follows BOS,
"cat" often follows "the", verbs follow nouns — and the loss drops. Here is
what each milestone roughly means:

| Loss | Probability on correct word | What's happening |
|---|---|---|
| 6.39 | 0.17% (1/597) | Random guessing — no knowledge |
| 3.22 | 4.0% | Learning common words and sentence starters |
| 2.85 | 5.8% | Picking up basic word-order patterns |
| 2.40 | 9.1% | Noun-verb associations forming |
| 1.76 | 17.2% | Solid grasp of sentence structure |

A loss of ~1.76 means the model is, on average, assigning about `e^(-1.76) ≈
17%` probability to the correct next word. That may sound low, but remember:
it is choosing among 597 words. Assigning 17% to the right answer means it has
narrowed the field from 597 equally likely candidates to roughly 6 plausible
ones. That is a 100x improvement over where it started.

## Where the Learning Happens

The steepest drop occurs in the first 1,000 steps. This is when the model
learns the easiest patterns: which words are common, which words tend to start
sentences, and basic category associations (articles before nouns, nouns before
verbs). The later steps produce smaller improvements as the model works on
harder patterns — less frequent words, longer-range dependencies, and the
subtle differences between similar sentence structures.

This is typical of neural network training: easy patterns are learned first,
and each subsequent improvement is harder to achieve than the last.
