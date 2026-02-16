# Backward Pass: The Chain Rule

The `backward()` method answers this question. Starting from the loss (with
gradient 1.0), it walks backward through every node in the graph, applying
the chain rule:

```
gradient of child += (gradient of parent) * (local gradient)
```

The `+=` is critical: when a Value feeds into multiple operations, its gradient
must sum contributions from all paths. Using `=` would overwrite earlier
contributions and produce incorrect gradients.

```typescript
backward(): void {
  const topo: Value[] = [];
  const visited = new Set<Value>();
  const buildTopo = (v: Value): void => {
    if (!visited.has(v)) {
      visited.add(v);
      for (const child of v.children) buildTopo(child);
      topo.push(v);
    }
  };
  buildTopo(this);
  this.grad = 1;
  for (const v of topo.reverse()) {
    for (let i = 0; i < v.children.length; i++) {
      v.children[i].grad += v.localGrads[i] * v.grad;
    }
  }
}
```

First it sorts the graph topologically (inputs before outputs), then walks it
in reverse. For each node, it multiplies its own gradient by each local
gradient and accumulates it onto the child's gradient.

After `backward()` runs, every `Value` in the entire computation, including
all model parameters, has its `.grad` field filled in. Each gradient says:
"increase this parameter slightly and the loss changes by this much."

Note that `backward()` does not zero out gradients before running — it sets
`this.grad = 1` at the root and accumulates onto whatever `.grad` values
already exist. Calling it twice on the same graph would double-count every
gradient. In our training loop this is not a problem: each step builds a fresh
computation graph, and we explicitly reset every parameter's gradient to zero
before the next step.

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
