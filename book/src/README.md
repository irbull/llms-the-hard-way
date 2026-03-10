# LLMs, the Hard Way

<div style="text-align: center;">
  <img src="hard-way-icon.png" alt="LLMs, the Hard Way" style="width: 200px;" />
</div>

Build a GPT language model from scratch in TypeScript. No frameworks, no
libraries, just math.

This tutorial walks you through every piece of a working language model: from
tokenization to attention to training to generation. By the end you will have
trained a tiny GPT that writes grade-school sentences and can be fine-tuned to
generate questions.

The code is intentionally minimal. The entire model fits in a few hundred lines
of TypeScript. Every operation, every matrix multiply, every gradient, is
written by hand so you can see exactly what happens inside a transformer.

## What you will build

- A **tokenizer** that maps words to integers
- An **autograd engine** that computes gradients automatically
- **Neural network primitives**: linear layers, softmax, normalization
- A **transformer model** with attention, MLPs, and residual connections
- A **training loop** with the Adam optimizer
- **Text generation** with temperature, top-k, and top-p sampling
- **Fine-tuning** to adapt the model to a new task

## How to read this tutorial

The tutorial is split into three parts:

1. **Building the Model**: the components that make up a transformer
2. **Training and Inference**: teaching the model and using it
3. **Putting It to Work**: end-to-end training, generation, and fine-tuning

Each section builds on the last. The code is cumulative: by the final chapter
you have a complete, working system.
