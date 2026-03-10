# The Computation Graph

As the model processes a token, every addition, multiplication, exponentiation
and so on creates a new Value node. The result is a **computation graph**, a
tree-like structure connecting the parameters at the leaves to a single
loss value at the root.

## Building the Graph

Consider a tiny example using the Value operations from the previous section:

```typescript
const x = new Value(2.0);
const y = new Value(3.0);
const z = x.mul(y);          // z = 6.0
const loss = z.add(1);       // loss = 7.0
```

Each operation creates a new Value that records its children and local
gradients. After these four lines, the graph looks like this:

```
x(2.0) ──┐
          mul ──→ z(6.0) ──┐
y(3.0) ──┘                 add ──→ loss(7.0)
                  1.0  ────┘
```

And the internal linkages are:

| Node | `.data` | `.children` | `.localGrads` |
|------|---------|-------------|---------------|
| `x` | 2.0 | [] | [] |
| `y` | 3.0 | [] | [] |
| `z` | 6.0 | [x, y] | [3.0, 2.0] |
| `loss` | 7.0 | [z, Value(1)] | [1, 1] |

Leaf nodes (`x` and `y`) have no children; they are the inputs. The `mul`
node records both inputs as children and stores `[y.data, x.data]` as local
gradients (the product rule). The `add` node stores `[1, 1]` because addition
passes gradients through unchanged.

This is the entire data structure that `backward()` will walk. No separate
graph object, no registration step. The graph is just the chain of `.children`
pointers from the loss back to the leaves.

## From Loss to Parameters

In the real model, the graph is much larger (thousands of nodes) but the
structure is the same. The `loss` at the root points to the softmax outputs,
which point to the logits, which point to the attention and MLP operations,
which eventually point to the embedding lookups and weight matrices. Every
parameter is reachable by following `.children` pointers from the loss.

The `backward()` method traverses this graph in reverse (from loss to leaves),
using each node's `.localGrads` to propagate the gradient signal. That is the
subject of the next section.
