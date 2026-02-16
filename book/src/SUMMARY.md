# Summary

[Introduction](README.md)

# Building the Model

- [Prerequisites](01-prerequisites/README.md)
    - [Software](01-prerequisites/software.md)
    - [The Training Data](01-prerequisites/training-data.md)
    - [The Pipeline](01-prerequisites/pipeline.md)

- [The Tokenizer](02-tokenizer/README.md)
    - [Building the Vocabulary](02-tokenizer/vocabulary.md)
    - [Encoding and Decoding](02-tokenizer/encoding-decoding.md)
    - [Why Word-Level Tokens?](02-tokenizer/why-word-level.md)
    - [Complete Code](02-tokenizer/complete-code.md)

- [The Autograd Engine](03-autograd/README.md)
    - [The Math You Need](03-autograd/the-math.md)
    - [The Value Class](03-autograd/value-class.md)
    - [Derived Operations](03-autograd/derived-operations.md)
    - [The Computation Graph](03-autograd/computation-graph.md)
    - [Backward Pass](03-autograd/backward-pass.md)
    - [Complete Code](03-autograd/complete-code.md)

- [Neural Network Primitives](04-nn-primitives/README.md)
    - [Linear](04-nn-primitives/linear.md)
    - [Softmax](04-nn-primitives/softmax.md)
    - [RMSNorm](04-nn-primitives/rmsnorm.md)
    - [Complete Code](04-nn-primitives/complete-code.md)

- [The Model](05-model/README.md)
    - [Configuration](05-model/configuration.md)
    - [Parameters](05-model/parameters.md)
    - [Embeddings](05-model/embeddings.md)
    - [Attention](05-model/attention.md)
    - [MLP](05-model/mlp.md)
    - [Residual Connections](05-model/residual-connections.md)
    - [The KV Cache](05-model/kv-cache.md)
    - [Creating the Model](05-model/creating-the-model.md)
    - [Running the Model](05-model/running-the-model.md)
    - [Complete Code](05-model/complete-code.md)

# Training and Inference

- [Training](06-training/README.md)
    - [The Training Configuration](06-training/configuration.md)
    - [The Training Loop](06-training/step-by-step.md)
    - [Watching It Learn](06-training/watching-it-learn.md)
    - [Complete Code](06-training/complete-code.md)

- [Saving the Model](07-saving/README.md)
    - [What Gets Saved](07-saving/what-gets-saved.md)
    - [Loading the Model](07-saving/loading.md)

- [Generation](08-generation/README.md)
    - [The Generation Loop](08-generation/generation-loop.md)
    - [Sampling Strategies](08-generation/sampling-strategies.md)
    - [The KV Cache](08-generation/kv-cache.md)
    - [Example Output](08-generation/example-output.md)
    - [Complete Code](08-generation/complete-code.md)

# Putting It to Work

- [Smoke Test](09-smoke-test/README.md)
    - [Train the Model](09-smoke-test/train.md)
    - [Generate Sentences](09-smoke-test/generate.md)
    - [What You Have Built](09-smoke-test/what-you-built.md)
    - [Complete Code](09-smoke-test/complete-code.md)

- [Fine-Tuning](10-fine-tuning/README.md)
    - [The Question Dataset](10-fine-tuning/question-dataset.md)
    - [Run the Fine-Tuning](10-fine-tuning/run-fine-tuning.md)
    - [Generate Questions](10-fine-tuning/generate-questions.md)
    - [Catastrophic Forgetting](10-fine-tuning/catastrophic-forgetting.md)
    - [Complete Code](10-fine-tuning/complete-code.md)
