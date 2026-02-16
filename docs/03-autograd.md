# Lab 03: The Autograd Engine

Training a neural network requires answering one question for each of the
model's parameters: **if I nudge this number up a tiny bit, does the model get
better or worse?**

The answer is the parameter's **gradient**: the direction and magnitude of
change that would reduce the model's error. Computing gradients by hand for
tens of thousands of interconnected parameters would be impossible. Instead, we
use **automatic differentiation**: we build a record of every computation the
model performs, then trace backwards through that record to compute all
gradients at once.

## The Value Class: A Number That Remembers

Every number in the model is not a plain `number`. It is a `Value`. A Value
stores three things:

1. **`data`**: the actual number (e.g., 0.03)
2. **`grad`**: the gradient, filled in later during the backward pass
3. **`_children` and `_localGrads`**: how this Value was computed from other Values

When we perform any operation on Values, the result is a new Value that
remembers its parents and the **local gradients**, the partial derivatives of
that operation with respect to each input. These local gradients answer: "if I
nudge this input, how much does the output change?"

A quick refresher: the **derivative** of a function is its **rate of change**
: how fast the output is changing relative to the input at a given point. If
f(3) = 9 and f(3.001) = 9.006, then the derivative at 3 is roughly
0.006 / 0.001 = 6. It is the slope of the function at that point.

Every primitive operation in our engine records these derivatives. Here they
all are, with the intuition for why each gradient is what it is.

### Addition: `a + b` (local gradients `[1, 1]`)

```typescript
add(other: Value | number): Value {
  const o = typeof other === "number" ? new Value(other) : other;
  return new Value(this.data + o.data, [this, o], [1, 1]);
}
```

- d(a + b) / da = **1**
- d(a + b) / db = **1**

If you nudge `a` up by 0.001, the sum goes up by exactly 0.001. The output
changes by the same amount as the input, regardless of what the values are.
Hence `[1, 1]`.

| a | b | a + b | Nudge a to 3.001 | New result | Change | Local grad |
|---|---|-------|-------------------|------------|--------|------------|
| 3 | 5 | 8     | 3.001 + 5         | 8.001      | 0.001  | 1          |

### Multiplication: `a * b` (local gradients `[b, a]`)

```typescript
mul(other: Value | number): Value {
  const o = typeof other === "number" ? new Value(other) : other;
  return new Value(this.data * o.data, [this, o], [o.data, this.data]);
}
```

- d(a * b) / da = **b**
- d(a * b) / db = **a**

If you nudge `a` up by 0.001, the product goes up by `0.001 * b`. The
sensitivity to `a` depends on how large `b` is, and vice versa. Hence
`[o.data, this.data]`, meaning the gradient for each input is the *other* input's
value.

| a | b | a * b | Nudge a to 3.001 | New result | Change | Local grad |
|---|---|-------|-------------------|------------|--------|------------|
| 3 | 5 | 15    | 3.001 * 5         | 15.005     | 0.005  | 5 (= b)    |

### Power: `a ^ n` (local gradient `[n * a^(n-1)]`)

```typescript
pow(n: number): Value {
  return new Value(this.data ** n, [this], [n * this.data ** (n - 1)]);
}
```

- d(a^n) / da = **n * a^(n-1)**

This is the classic power rule from calculus. The exponent drops down as a
coefficient, and the power decreases by one. For `a^2`, the gradient is `2a`.
The larger `a` is, the more sensitive the square is to small changes.

| a | n | a ^ n | Nudge a to 3.001 | New result | Change | Local grad     |
|---|---|-------|-------------------|------------|--------|----------------|
| 3 | 2 | 9     | 3.001^2           | 9.006001   | ~0.006 | 6 (= 2 * 3)   |

### Log: `ln(a)` (local gradient `[1 / a]`)

```typescript
log(): Value {
  return new Value(Math.log(this.data), [this], [1 / this.data]);
}
```

- d(ln(a)) / da = **1 / a**

The log function is steep when `a` is small and flat when `a` is large. A tiny
nudge to a small number produces a big change in the log; the same nudge to a
large number barely moves it.

| a    | ln(a)  | Nudge a to 0.101 | New result | Change | Local grad       |
|------|--------|------------------|------------|--------|------------------|
| 0.1  | -2.303 | ln(0.101)        | -2.293     | ~0.010 | 10 (= 1 / 0.1)  |
| 10.0 | 2.303  | ln(10.001)       | 2.3026     | ~0.0001| 0.1 (= 1 / 10)  |

### Exp: `e^a` (local gradient `[e^a]`)

```typescript
exp(): Value {
  return new Value(Math.exp(this.data), [this], [Math.exp(this.data)]);
}
```

- d(e^a) / da = **e^a**

The exponential function is its own derivative. The larger the output is, the
faster it grows. A nudge to `a` changes the output by an amount proportional
to the output itself.

| a | e^a    | Nudge a to 2.001 | New result | Change | Local grad       |
|---|--------|------------------|------------|--------|------------------|
| 2 | 7.389  | e^2.001          | 7.396      | ~0.007 | 7.389 (= e^2)   |

### ReLU: `max(0, a)` (local gradient `[1 if a > 0, else 0]`)

```typescript
relu(): Value {
  return new Value(Math.max(0, this.data), [this], [this.data > 0 ? 1 : 0]);
}
```

- d(relu(a)) / da = **1** if a > 0, **0** if a <= 0

ReLU is the simplest nonlinearity: it passes positive values through unchanged
and clamps negatives to zero. When `a` is positive, the gradient is 1 (the
nudge passes through). When `a` is negative, the gradient is 0 (the output is
stuck at zero, so nudging `a` does nothing).

| a  | relu(a) | Nudge a by 0.001 | New result | Change | Local grad |
|----|---------|------------------|------------|--------|------------|
| 3  | 3       | relu(3.001)      | 3.001      | 0.001  | 1          |
| -2 | 0       | relu(-1.999)     | 0          | 0      | 0          |

### Derived Operations

The remaining operations (`neg`, `sub`, `div`) don't need their own gradient
rules. They are composed from the primitives above:

- **`neg(a)`** = `a * (-1)`, uses `mul`
- **`sub(a, b)`** = `a + (-b)`, uses `add` and `neg`
- **`div(a, b)`** = `a * b^(-1)`, uses `mul` and `pow`

Because every step is tracked in the computation graph, the chain rule handles
these compositions automatically.

## The Computation Graph

As the model processes a token, every addition, multiplication, exponentiation
and so on creates a new Value node. The result is a **computation graph**, a
tree-like structure connecting the parameters at the leaves to a single
loss value at the root.

Here is a tiny fragment of what this graph looks like:

```
param_0 --mul--+
               add-- ... -- loss
param_1 --mul--+
```

The `loss` tells us how wrong the model's prediction was. We want to know:
for each parameter, which direction should we nudge it to make the loss
smaller?

## Backward Pass: The Chain Rule

The `backward()` method answers this question. Starting from the loss (with
gradient 1.0), it walks backward through every node in the graph, applying
the chain rule:

```
gradient of child = (gradient of parent) * (local gradient)
```

```typescript
backward(): void {
  const topo: Value[] = [];
  const visited = new Set<Value>();
  const buildTopo = (v: Value): void => {
    if (!visited.has(v)) {
      visited.add(v);
      for (const child of v._children) buildTopo(child);
      topo.push(v);
    }
  };
  buildTopo(this);
  this.grad = 1;
  for (const v of topo.reverse()) {
    for (let i = 0; i < v._children.length; i++) {
      v._children[i].grad += v._localGrads[i] * v.grad;
    }
  }
}
```

First it sorts the graph topologically (parents after children), then walks it
in reverse. For each node, it multiplies its own gradient by each local
gradient and accumulates it onto the child's gradient.

After `backward()` runs, every `Value` in the entire computation, including
all model parameters, has its `.grad` field filled in. Each gradient says:
"increase this parameter slightly and the loss changes by this much."

## The vsum Helper

One more utility: `vsum` adds a list of Values through the computation graph,
so the sum is also differentiable:

```typescript
export function vsum(values: Value[]): Value {
  return values.reduce((acc, v) => acc.add(v), new Value(0));
}
```

This is used throughout: for dot products, for summing losses, and for
normalization.

Next: [Lab 04: Neural Network Primitives](04-nn-primitives.md)
