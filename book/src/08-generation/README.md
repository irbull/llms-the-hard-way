# Generation

The model is trained. The weights are saved. Now we load them and generate new
sentences.

Generation works one word at a time. Start with the BOS token, feed it to the
model, get 597 scores back, pick a word, feed that word back in, and repeat.
Each step asks the same question: **given everything so far, what word comes
next?** The output of one step becomes the input to the next — this is called
**autoregressive** generation.

The interesting problem is how to pick the next word. The model outputs 597 raw
scores (logits), but choosing the highest-scoring word every time produces dull,
repetitive text. Choosing completely at random produces nonsense. The sampling
strategy — temperature, top-k, and top-p — controls where the output lands
between those extremes.

```
BOS → model → 597 logits → temperature → top-k → top-p → softmax → sample → next token
 ↑                                                                              |
 └────────────────────────────────────────────────────────────────────────────────┘
```

This chapter covers the generation loop, the KV cache that makes it efficient,
and the sampling knobs that shape the output.
