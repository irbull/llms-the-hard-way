# The Pipeline

Here is the path from raw text to a model that generates new sentences:

```
training data  ->  tokenizer  ->  model  ->  training  ->  saved weights
                                                               |
                   tokenizer  ->  model  ->  generation  <-  load weights
```

We build each piece from scratch, in order. Each chapter introduces one
component and ends with a **Complete Code** page containing the finished
source for that stage.

## What You Will Build

| Chapter | You will create | Pipeline stage |
|---|---|---|
| [The Tokenizer](../02-tokenizer/README.md) | `tokenizer.ts` | Turns words into numbers and back |
| [The Autograd Engine](../03-autograd/README.md) | `autograd.ts` | Automatic differentiation (makes training possible) |
| [Neural Network Primitives](../04-nn-primitives/README.md) | `nn.ts` | Linear layers, softmax, normalization |
| [The Model](../05-model/README.md) | `model.ts`, `rng.ts` | The GPT architecture: config, weights, forward pass |
| [Training](../06-training/README.md) | `train.ts` | The training loop and optimizer |
| [Saving the Model](../07-saving/README.md) | `saveModel`, `loadModel` | Serialize trained weights to disk and load them back |
| [Generation](../08-generation/README.md) | `generate.ts` | Inference: turning a trained model into sentences |
| [Smoke Test](../09-smoke-test/README.md) | `phrases-train.ts`, `phrases-generate.ts` | Entry points to train and generate |
| [Fine-Tuning](../10-fine-tuning/README.md) | `phrases-fine-tune.ts` | Adapt a trained model to new data |
