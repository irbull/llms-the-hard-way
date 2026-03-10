# RMSNorm (Normalization)

Rescales a vector to have roughly unit variance. This prevents numbers from
growing too large or too small as they pass through multiple layers:

```typescript
function rmsnorm(input: Value[]): Value[] {
  const ms = vsum(input.map((xi) => xi.mul(xi))).div(input.length);
  const scale = ms.add(1e-5).pow(-0.5);
  return input.map((xi) => xi.mul(scale));
}
```

It computes the root mean square of the vector, then divides each element by
it. The `1e-5` prevents division by zero. For example:

```
input = [3.0, 4.0]

mean of squares:  (9 + 16) / 2 = 12.5
root mean square: √12.5 ≈ 3.54
divide each:      [3.0 / 3.54, 4.0 / 3.54] ≈ [0.85, 1.13]
```

The values have been rescaled so they sit near 1, but their relative
proportions (3:4) are preserved. Without normalization, activations can
explode or vanish across layers, making training unstable.

Real RMSNorm includes a learnable per-element scale parameter (gamma) that
lets each dimension adjust its magnitude after normalization. We omit it for
simplicity; our model trains fine without it at this scale.

## Summary

These three primitives (`linear`, `softmax`, and `rmsnorm`) are the
building blocks the model assembles in the next chapter. Each one is built entirely from
`Value` operations, so the autograd engine can compute gradients through all
of them.

| Primitive | What it does | Where it is used |
|---|---|---|
| `linear` | Matrix-vector multiply | Attention projections, MLP layers, output head |
| `softmax` | Scores to probabilities | Attention weights, training loss, generation sampling |
| `rmsnorm` | Normalize to unit variance | Before each transformer sub-block |
