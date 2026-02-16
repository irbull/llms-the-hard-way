# Lab 01: Prerequisites

In this lab you will set up the tools needed to complete this tutorial and
examine the training data that the model will learn from.

## Software

This tutorial requires [Node.js](https://nodejs.org/) (v18 or later) and
[TypeScript](https://www.typescriptlang.org/). Verify they are installed:

```bash
node --version
```

> v22.15.0

## Clone the Repository

Clone the tutorial repository and install dependencies:

```bash
git clone https://github.com/user/ai-the-hard-way.git
cd ai-the-hard-way
npm install
```

## The Training Data

Every language model starts with a question: **what do we want it to learn?**

We want ours to learn how simple English sentences are put together. The
training corpus is 30,000 grade-1-level sentences in
`data/grade1_sentences.txt`:

```bash
wc -l data/grade1_sentences.txt
```

> 30000 data/grade1_sentences.txt

Look at a few:

```bash
head -10 data/grade1_sentences.txt
```

```
nan has the nut
the shy bed is old
the bird is wide
the goat likes to go
the cow yells
we dance up the garden
we swing up the store
the cat eats a muffin
the hen walks at dusk
the bat digs to the hill
```

Each sentence is short (2-8 words), uses basic vocabulary, and follows simple
subject-verb-object patterns. The entire vocabulary is only **596 unique
words**, a fraction of what a real LLM would handle, but enough to see every
concept in action.

## What the Model Will Learn

We will never tell the model anything about English grammar. We will not define
"noun" or "verb" or "sentence." We will simply show it thousands of sentences
and ask: **given the words so far, what word comes next?**

That is all a language model does. It learns to predict the next token. And
from this single task, next-token prediction, structure emerges: the model
learns that "the" is often followed by a noun, that sentences end after a
handful of words, that certain words cluster together.

## The Pipeline

Here is the path from raw text to a model that generates new sentences:

```
training data  ->  tokenizer  ->  model  ->  training  ->  saved weights
                                                               |
                   tokenizer  ->  model  ->  generation  <-  load weights
```

We build each piece from scratch, in order.

## Source Files

| File | What it does |
|---|---|
| `data/grade1_sentences.txt` | 30,000 training sentences |
| `src/tokenizer.ts` | Turns words into numbers and back |
| `src/autograd.ts` | Automatic differentiation (makes training possible) |
| `src/nn.ts` | Neural net primitives: linear layers, softmax, normalization |
| `src/rng.ts` | Seeded random number generator |
| `src/model.ts` | The GPT architecture: config, weights, forward pass |
| `src/train.ts` | The training loop and optimizer |
| `src/generate.ts` | Inference: turning a trained model into sentences |
| `src/phrases-train.ts` | Entry point: train and save the model |
| `src/phrases-generate.ts` | Entry point: load the model and generate |

Next: [Lab 02: The Tokenizer](02-tokenizer.md)
