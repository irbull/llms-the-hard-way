# Linear (Matrix-Vector Multiply)

The fundamental neural net operation. It transforms a vector by multiplying it
with a weight matrix. First, a type alias. A `Matrix` is just a 2D array of
Values:

```typescript
type Matrix = Value[][];
```

```typescript
function linear(input: Value[], weights: Matrix): Value[] {
  return weights.map((row) => vsum(row.map((w, i) => w.mul(input[i]))));
}
```

Each output element is a **dot product**: multiply each input by the
corresponding weight, then sum the results. Here is a concrete example with a
3-element input and a 2×3 weight matrix:

```
input = [2, 3, 1]

weights = [[1, 0, -1],
           [0, 2,  1]]

output[0] = (1×2) + (0×3) + (-1×1) = 1
output[1] = (0×2) + (2×3) + ( 1×1) = 7

output = [1, 7]
```

Each row of the weight matrix produces one output element. A weight matrix
with shape 2×3 takes a 3-element input and produces a 2-element output. In the
actual model, a weight matrix with shape 128×32 takes a 32-element input
vector and produces a 128-element output vector. The operation is the same,
just more rows and longer dot products.

This is how the model mixes information across dimensions.

Because every multiplication and addition uses `Value` nodes, the entire
operation is differentiable, so gradients flow back through it during training.
