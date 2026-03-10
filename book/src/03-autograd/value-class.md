# The Value Class: A Number That Remembers

Every number in the model is not a plain `number`. It is a `Value`. A Value
stores three things:

1. **`data`**: the actual number (e.g., 0.03)
2. **`grad`**: the gradient, filled in later during the backward pass
3. **`children` and `localGrads`**: how this Value was computed from other Values

When we perform any operation on Values, the result is a new Value that
remembers its **children** (the input Values) and the **local gradients**, the
partial derivatives of that operation with respect to each input. These local gradients answer: "if I
nudge this input, how much does the output change?"

`children` and `localGrads` are parallel arrays: one local gradient per
child. A binary operation like `add` or `mul` takes two inputs, so both arrays
have two entries. A unary operation like `pow`, `log`, or `relu` takes one
input, so both arrays have one entry. The backward pass pairs them up: the
gradient for `children[i]` is computed using `localGrads[i]`.

Here is the constructor:

```typescript
export class Value {
  data: number;
  grad: number;
  children: Value[];
  localGrads: number[];

  constructor(data: number, children: Value[] = [], localGrads: number[] = []) {
    this.data = data;
    this.grad = 0;
    this.children = children;
    this.localGrads = localGrads;
  }
```

A leaf Value (like a model parameter) is created with just a number:
`new Value(0.03)`. The `children` and `localGrads` default to empty arrays.
When an operation creates a new Value, it passes in the inputs and their local
gradients.

Every primitive operation in our engine records these derivatives. Here they
all are, with the intuition for why each gradient is what it is.

## Addition: `a + b` (local gradients `[1, 1]`)

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

## Multiplication: `a * b` (local gradients `[b, a]`)

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

## Power: `a ^ n` (local gradient `[n * a^(n-1)]`)

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

## Log: `ln(a)` (local gradient `[1 / a]`)

```typescript
log(): Value {
  return new Value(Math.log(this.data), [this], [1 / this.data]);
}
```

- d(ln(a)) / da = **1 / a**

The log function is steep when `a` is small and flat when `a` is large. A tiny
nudge to a small number produces a big change in the log; the same nudge to a
large number barely moves it. Note: `log(0)` is `-Infinity` and the gradient
`1/0` poisons the computation. In our model, `log()` is only called on softmax
outputs, which are always positive, so this is safe in practice.

| a    | ln(a)  | Nudge a to 0.101 | New result | Change | Local grad       |
|------|--------|------------------|------------|--------|------------------|
| 0.1  | -2.303 | ln(0.101)        | -2.293     | ~0.010 | 10 (= 1 / 0.1)  |
| 10.0 | 2.303  | ln(10.001)       | 2.3026     | ~0.0001| 0.1 (= 1 / 10)  |

## Exp: `e^a` (local gradient `[e^a]`)

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

## ReLU: `max(0, a)` (local gradient `[1 if a > 0, else 0]`)

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
