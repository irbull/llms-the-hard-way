# LLMs, the Hard Way

I really appreciate it when someone distills complex concepts down to their
building blocks. Nobody does this better than [Kelsey Hightower](https://github.com/kelseyhightower) and [Andrej
Karpathy](https://github.com/karpathy). I've taken inspiration from them and built **LLMs, the Hard Way**, a
ground-up guide to building a language model that writes English sentences. No frameworks. No
matrix libraries. Just TypeScript, math, and 30,000 sentences a first-grader
could read.

This tutorial was inspired by [Karpathy's MicroGPT](https://karpathy.github.io/2026/02/12/microgpt/) with a slightly
different dataset. Instead of human names, I used grade school sentences. I
changed the tokenizer to use words instead of letters and split the process
into a separate Train and Inference loop. I then structured the tutorial like
[Hightower's Kubernetes The Hard Way](https://github.com/kelseyhightower/kubernetes-the-hard-way).

By the end you will have trained a tiny GPT that produces sentences like:

```
the cat runs to the park
mom reads a big book
seven ducks swim
```

Not because anyone told it the rules of English, but because it learned the
patterns from data.

The architecture here is the same one behind real large language models --
the same tokenizer, embeddings, attention mechanism, transformer layers, and
training loop. The only differences are scale: a 597-word vocabulary instead of
hundreds of thousands, 63,000 parameters instead of billions, and 30,000
training sentences instead of trillions of tokens. Everything else is just
scaling and optimization.

## Labs

- [Lab 01: Prerequisites](docs/01-prerequisites.md) - Set up tools and examine
  the training data
- [Lab 02: The Tokenizer](docs/02-tokenizer.md) - Turn words into numbers and
  back
- [Lab 03: The Autograd Engine](docs/03-autograd.md) - Build the automatic
  differentiation system
- [Lab 04: Neural Network Primitives](docs/04-nn-primitives.md) - Linear layers,
  softmax, normalization
- [Lab 05: The Model](docs/05-model.md) - Embeddings, transformer layers, and
  the forward pass
- [Lab 06: Training](docs/06-training.md) - Cross-entropy loss, backpropagation,
  and the Adam optimizer
- [Lab 07: Saving the Model](docs/07-saving.md) - Serialize the trained weights
  to disk
- [Lab 08: Generation](docs/08-generation.md) - Temperature, top-k, top-p, and
  the KV cache
- [Lab 09: Smoke Test](docs/09-smoke-test.md) - Train, generate, and verify the
  output
