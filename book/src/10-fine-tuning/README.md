# Fine-Tuning

Our model generates declarative sentences: "the cat runs to the park." But what
if we want it to generate questions instead? We could train a new model from
scratch on question data, but there is a much better way: **fine-tuning**.

Fine-tuning means taking a model that already knows a language and continuing to
train it on a small, specialized dataset. The model keeps its existing knowledge
(what words mean, how sentences are structured) and learns a new pattern (how
to form questions) on top of it.

This is exactly how ChatGPT works. OpenAI first trains a base model on trillions
of tokens of internet text (pre-training), then fine-tunes it on a much smaller
dataset of conversations (fine-tuning). The base model learns language; the
fine-tuning teaches it to be a helpful assistant.
