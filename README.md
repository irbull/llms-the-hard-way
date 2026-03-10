<p align="center">
  <img src="hard-way-icon.png" alt="LLMs, the Hard Way" width="128" height="128">
</p>

# LLMs, the Hard Way: https://llms.ianbull.com

I really appreciate it when someone distills complex concepts down to their
building blocks. Nobody does this better than [Kelsey Hightower](https://github.com/kelseyhightower) and [Andrej
Karpathy](https://github.com/karpathy). I've taken inspiration from them and built **LLMs, the Hard Way**, a
ground-up guide to building a language model that writes English sentences. No frameworks. No
matrix libraries. Just TypeScript, math, and 30,000 sentences a first-grader
could read.

The tutorial is published as a mdBook:
### [https://llms.ianbull.com](https://llms.ianbull.com)

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

The tutorial is written as an [mdbook](https://rust-lang.github.io/mdBook/).
Each lab builds on the previous one, with complete code at each step in `solutions/`.

1. **[Prerequisites](book/src/01-prerequisites/README.md)** - Set up tools and examine the training data
2. **[The Tokenizer](book/src/02-tokenizer/README.md)** - Turn words into numbers and back
3. **[The Autograd Engine](book/src/03-autograd/README.md)** - Build the automatic differentiation system
4. **[Neural Network Primitives](book/src/04-nn-primitives/README.md)** - Linear layers, softmax, normalization
5. **[The Model](book/src/05-model/README.md)** - Embeddings, transformer layers, and the forward pass
6. **[Training](book/src/06-training/README.md)** - Cross-entropy loss, backpropagation, and the Adam optimizer
7. **[Saving the Model](book/src/07-saving/README.md)** - Serialize the trained weights to disk
8. **[Generation](book/src/08-generation/README.md)** - Temperature, top-k, top-p, and the KV cache
9. **[Smoke Test](book/src/09-smoke-test/README.md)** - Train, generate, and verify the output
10. **[Fine-Tuning](book/src/10-fine-tuning/README.md)** - Adapt the pre-trained model to generate questions
