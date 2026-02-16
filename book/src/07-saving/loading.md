# Loading the Model

To load, we recreate the model structure from the config, then fill in the
learned weights:

```typescript
export function loadModel(path: string): Model {
  const data = JSON.parse(readFileSync(path, "utf-8"));
  const model = createModel(data.config);
  for (let i = 0; i < model.params.length; i++) {
    model.params[i].data = data.weights[i];
  }
  return model;
}
```

`createModel(data.config)` allocates all the weight matrices with random values
(just like before training), then we overwrite every parameter with the learned
value. The ordering is deterministic (`createModel` always creates matrices in
the same order), so the flat weights array maps back to the correct matrices.

## Why This Matters

This separation (train once, generate many times) is how all production LLMs
work. Training a large model can cost hundreds of millions of dollars in
compute. But once trained, the model weights are saved and can be loaded for
inference on any machine. The training code, the optimizer state, the gradient
buffers: none of that is needed for inference. Just the architecture and the
numbers.

```
Training (expensive, done once):
  train  ->  phrases-model.json

Inference (cheap, done many times):
  phrases-model.json  ->  generate  ->  sentences
```

We will wire up these scripts in the smoke test chapter.
