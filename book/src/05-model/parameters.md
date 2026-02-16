# Parameters: The Model's Memory

When we create a model, we allocate a set of weight matrices filled with small
random numbers:

```typescript
function matrix(nout: number, nin: number, std = 0.08): Matrix {
  return Array.from({ length: nout }, () =>
    Array.from({ length: nin }, () => new Value(gauss(0, std)))
  );
}
```

Each weight matrix serves a specific role. Here is every matrix in our model
and what it does:

| Matrix | Shape | Purpose |
|---|---|---|
| `weights.tokenEmbedding` | 597 x 32 | **Token embeddings**: one 32-dim vector per word |
| `weights.positionEmbedding` | 16 x 32 | **Position embeddings**: one 32-dim vector per position |
| `weights.output` | 597 x 32 | **Output projection**: maps back to vocabulary |
| `layers[0].attention.query` | 32 x 32 | Attention query weights (layer 0) |
| `layers[0].attention.key` | 32 x 32 | Attention key weights (layer 0) |
| `layers[0].attention.value` | 32 x 32 | Attention value weights (layer 0) |
| `layers[0].attention.output` | 32 x 32 | Attention output weights (layer 0) |
| `layers[0].mlp.hidden` | 128 x 32 | MLP hidden layer (layer 0) |
| `layers[0].mlp.output` | 32 x 128 | MLP output layer (layer 0) |
| `layers[1].attention.query` | 32 x 32 | Attention query weights (layer 1) |
| `layers[1].attention.key` | 32 x 32 | Attention key weights (layer 1) |
| `layers[1].attention.value` | 32 x 32 | Attention value weights (layer 1) |
| `layers[1].attention.output` | 32 x 32 | Attention output weights (layer 1) |
| `layers[1].mlp.hidden` | 128 x 32 | MLP hidden layer (layer 1) |
| `layers[1].mlp.output` | 32 x 128 | MLP output layer (layer 1) |

Total: **63,296 parameters**. Every one of these numbers will be adjusted
during training.
