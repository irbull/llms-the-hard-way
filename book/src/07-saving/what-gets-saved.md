# What Gets Saved

A model is just configuration (the shape) plus parameters (the learned
numbers). We save both as JSON:

```typescript
export function saveModel(model: Model, path: string): void {
  const data = {
    config: model.config,
    weights: model.params.map((p) => p.data),
  };
  writeFileSync(path, JSON.stringify(data));
}
```

The config tells us the architecture (`nLayer: 2, nEmbd: 32, ...`) and the
weights array holds all 63,296 parameter values as plain numbers. The file is
about 1.5 MB of JSON.
