# The Model

With the autograd engine and neural net primitives in hand, we can build the
model. What is a model, concretely?

**A model is a configuration plus a large collection of numbers (parameters).**

Before training, these numbers are random. After training, they encode
everything the model has learned about language. The architecture (how those
numbers are wired together) determines what the model *can* learn. The
training process determines what it *does* learn.

## What the Model Contains

Concretely, the model is a nested structure of weight matrices — arrays of
arrays of `Value` nodes. Each matrix has a specific shape and role. Here is
every piece:

```
Model
├── config: { nLayer: 2, nEmbd: 32, vocabSize: 597, ... }
│
├── weights
│   ├── tokenEmbedding     [597 × 32]   one row per word in the vocabulary
│   ├── positionEmbedding  [16 × 32]    one row per position in a sentence
│   │
│   ├── layers[0]
│   │   ├── attention
│   │   │   ├── query      [32 × 32]    three identical projections
│   │   │   ├── key        [32 × 32]    (roles emerge from training,
│   │   │   ├── value      [32 × 32]     not from the code)
│   │   │   └── output     [32 × 32]    combine attention results
│   │   └── mlp
│   │       ├── hidden     [128 × 32]   expand to 128 dimensions
│   │       └── output     [32 × 128]   compress back to 32
│   │
│   ├── layers[1]                        (same structure as layer 0)
│   │
│   └── output             [597 × 32]   map back to vocabulary scores
│
└── params: all 63,296 Values in a flat array (for the optimizer)
```

Every number in every matrix is a `Value`. That means the autograd engine can
compute gradients for all 63,296 of them. The rest of this chapter explains
what each piece does and how they connect.

## The Forward Pass

Here is what happens when we feed a token into the model:

```
Token ID (e.g. 541 = "the") + Position (e.g. 2)
    |
[Token Embedding] + [Position Embedding]  ->  32-dim vector
    |
[RMSNorm]  ->  normalize the vector
    |
+--- Transformer Layer 0 ----------------------+
|  [RMSNorm]                                    |
|  [Multi-Head Attention]  ->  look at context  |
|  [+ Residual Connection]                      |
|  [RMSNorm]                                    |
|  [MLP: expand -> ReLU -> compress]  -> process|
|  [+ Residual Connection]                      |
+-----------------------------------------------+
    |
+--- Transformer Layer 1 ----------------------+
|  (same structure)                             |
+-----------------------------------------------+
    |
[output projection]  ->  597 raw scores (one per word)
    |
"logits", unnormalized predictions for the next word
```

The output is 597 raw scores, one per word in the vocabulary. These scores are
called **logits** — unnormalized numbers that can be positive, negative, large,
or small. A higher logit means the model thinks that word is more likely to
come next. On their own logits are not probabilities; they become probabilities
when passed through `softmax` (from the previous chapter) during training or generation.

Before training, logits are essentially random. After training, if you feed the
model "the cat", the logits for "runs," "eats," and "sits" will be much higher
than "the" or "zoo."

## Simplifications

Our architecture follows the same structure as GPT-2 and LLaMA, with a few
simplifications that keep the code short without changing how the model works:

| Our model | Standard GPT-2 / LLaMA | Why we simplify |
|---|---|---|
| RMSNorm | LayerNorm (GPT-2) / RMSNorm (LLaMA) | Fewer operations, same effect at this scale |
| ReLU activation | GELU (GPT-2) / SiLU (LLaMA) | Simpler gradient, easier to understand |
| No bias terms | Bias on every linear layer (GPT-2) | Fewer parameters, modern models drop them too |
| No learnable norm scale | Learnable gamma per element | One less thing to train, works fine here |
| No final norm before output | RMSNorm before output projection | Skipped for brevity |

None of these affect the core ideas. The architecture, the attention mechanism,
the training loop — all identical. A reader who understands this model
understands the real thing.
