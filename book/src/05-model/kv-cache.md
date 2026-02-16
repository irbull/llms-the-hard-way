# The KV Cache

The model processes **one token at a time**. If we have the sentence
"the cat eats...", when we feed it "cat" at position 2, it does not
automatically know that "the" came before it. But attention — the
mechanism that lets the model look at previous words — needs access to
those earlier tokens. How?

The answer is that we **cache** the key and value vectors computed at every
previous position. As we saw in the attention section, each token produces
three vectors: a query (Q), a key (K), and a value (V). The query is used
immediately and discarded. But the keys and values are needed again every time
a future token wants to attend to this position.

The KV cache is just a list that grows by one entry each time we process a
token:

```
After processing BOS at position 0:
  keys:   [ K_bos ]
  values: [ V_bos ]

After processing "the" at position 1:
  keys:   [ K_bos, K_the ]
  values: [ V_bos, V_the ]

After processing "cat" at position 2:
  keys:   [ K_bos, K_the, K_cat ]
  values: [ V_bos, V_the, V_cat ]
```

When the model processes "cat", its query vector is compared against **all**
cached keys (BOS, "the", and "cat" itself) to compute attention scores. Those
scores determine how much to weight each cached value vector. This is how
"cat" can attend to "the" even though "the" was processed in a previous call.

This is also how causal masking works in our implementation: the cache only
contains entries for tokens already processed, so there are no future keys
to attend to.

We create a fresh, empty cache at the start of each sentence:

```typescript
export function createKVCache(model: Model): {
  keys: Value[][][];
  values: Value[][][];
} {
  return {
    keys: Array.from({ length: model.config.nLayer }, () => []),
    values: Array.from({ length: model.config.nLayer }, () => []),
  };
}
```

The type `Value[][][]` is one list per layer (we have 2 layers, so 2 lists).
Each list starts empty and grows as tokens are processed. A fresh cache means
a fresh sentence — the model has no memory of previous sentences.
