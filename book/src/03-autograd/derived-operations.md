# Derived Operations

The remaining operations (`neg`, `sub`, `div`) don't need their own gradient
rules. They are composed from the primitives above:

- **`neg(a)`** = `a * (-1)`, uses `mul`
- **`sub(a, b)`** = `a + (-b)`, uses `add` and `neg`
- **`div(a, b)`** = `a * b^(-1)`, uses `mul` and `pow`

Because every step is tracked in the computation graph, the chain rule handles
these compositions automatically.
