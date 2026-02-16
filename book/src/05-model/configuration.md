# Configuration

The configuration defines the shape of the model:

```typescript
const model = createModel({
  nLayer: 2,       // number of transformer layers
  nEmbd: 32,       // embedding dimension (size of internal vectors)
  blockSize: 16,   // maximum sequence length (longest sentence we can process)
  nHead: 4,        // number of attention heads
  headDim: 8,      // dimension per attention head (nEmbd / nHead)
  vocabSize: 597,  // our tokenizer's vocabulary size
});
```

These are small numbers. Production models use `nEmbd` in the thousands and
dozens of layers. But the architecture is the same, ours just fits in memory
and trains in minutes instead of months.

A note on `nHead`: with 32 embedding dimensions, 4 heads is a good balance.
Each head gets `32 / 4 = 8` dimensions to work with. Two heads would give 16
dims each (fewer distinct attention patterns), and 8 heads would give 4 dims
each (very little room per head at this small scale).
