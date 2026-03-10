# The Training Data

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
nan mixed
the bird is wide
the goat likes to go
the cow yells
we dance up the garden
we swing up the store
the cat eats a muffin
the hen walks at dusk
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

## It's All Just Scale

Our model has 596 words, 2 layers, and around 63,000 parameters. A model
like GPT-4 has a vocabulary of 100,000+ tokens, nearly a hundred layers, and
over a trillion parameters. It trains on trillions of words scraped from
books, code repositories, scientific papers, and the open web.

But the architecture is the same. Embeddings, attention, feed-forward layers,
residual connections, softmax over the vocabulary: every piece you will build
in this tutorial is a piece that ships in production LLMs. The difference is
not a difference in kind. It is a difference in scale.

When a large model writes working code, answers a medical question, or
translates between languages, it is doing exactly what our tiny model does:
predicting the next token. It just has enough parameters and has seen enough
data that "predict the next token" becomes indistinguishable from
understanding.

There is no hidden trick. No secret module that "really" does the thinking.
The code you are about to write is the whole story.
