# Creating the Model

The `createModel` function allocates all the weight matrices and collects
every individual number into a flat `params` array:

```typescript
export function createModel(config: GPTConfig): Model {
  const weights: Weights = {
    tokenEmbedding: matrix(vocabSize, nEmbd),
    positionEmbedding: matrix(blockSize, nEmbd),
    output: matrix(vocabSize, nEmbd),
    layers: Array.from({ length: nLayer }, () => ({
      attention: {
        query: matrix(nEmbd, nEmbd),
        key: matrix(nEmbd, nEmbd),
        value: matrix(nEmbd, nEmbd),
        output: matrix(nEmbd, nEmbd),
      },
      mlp: {
        hidden: matrix(4 * nEmbd, nEmbd),
        output: matrix(nEmbd, 4 * nEmbd),
      },
    })),
  };

  // Collect all matrices into a flat param array
  const allMatrices: Matrix[] = [
    weights.tokenEmbedding,
    weights.positionEmbedding,
    weights.output,
    ...weights.layers.flatMap((layer) => [
      layer.attention.query, layer.attention.key,
      layer.attention.value, layer.attention.output,
      layer.mlp.hidden, layer.mlp.output,
    ]),
  ];
  const params = allMatrices.flatMap((mat) => mat.flatMap((row) => row));

  return { config, weights, params };
}
```

The `params` array is what the optimizer will update during training. The
`weights` object is a typed view into those same parameters. When training updates
`params[i]`, the corresponding entry in `weights` changes too, because they
are the same `Value` objects.

## Putting It All Together

Every operation (embedding lookup, `linear` transform, `softmax`,
`rmsnorm`, addition, ReLU) is built from `Value` nodes. The entire forward
pass builds one enormous computation graph. When we call `backward()` on the
loss, the gradients for all 63,296 parameters are computed in a single sweep
through this graph.

This is what makes neural network training possible: the autograd engine turns
the question "how should I change 63,296 numbers to make my predictions
better?" into a mechanical, automatic computation.

At this point we have 63,296 random numbers and a blueprint for how to wire
them together. The model can process tokens, but its output is nonsense. To
make it useful, we need to train it.
