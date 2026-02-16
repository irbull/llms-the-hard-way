# Lab 05: The Model

With the autograd engine and neural net primitives in hand, we can build the
model. What is a model, concretely?

**A model is a configuration plus a large collection of numbers (parameters).**

Before training, these numbers are random. After training, they encode
everything the model has learned about language. The architecture (how those
numbers are wired together) determines what the model *can* learn. The
training process determines what it *does* learn.

## Configuration

The configuration defines the shape of the model:

```typescript
const model = createModel({
  nLayer: 2,       // number of transformer layers
  nEmbd: 32,       // embedding dimension (size of internal vectors)
  blockSize: 16,   // maximum sequence length (longest sentence we can process)
  nHead: 4,        // number of attention heads
  headDim: 8,      // dimension per attention head (nEmbd / nHead)
  vocabSize: 597,  // our tokenizer's vocabulary size
});
```

These are small numbers. Production models use `nEmbd` in the thousands and
dozens of layers. But the architecture is the same, ours just fits in memory
and trains in minutes instead of months.

A note on `nHead`: with 32 embedding dimensions, 4 heads is really the only
sensible choice. Each head gets `32 / 4 = 8` dimensions to work with. Two
heads would give 16 dims each (too coarse to learn distinct patterns), and 8
heads would give 4 dims each (too little room per head).

## Parameters: The Model's Memory

When we create a model, we allocate a set of weight matrices filled with small
random numbers:

```typescript
function matrix(nout: number, nin: number, std = 0.08): Matrix {
  return Array.from({ length: nout }, () =>
    Array.from({ length: nin }, () => new Value(gauss(0, std)))
  );
}
```

Each weight matrix serves a specific role. Here is every matrix in our model
and what it does:

| Matrix | Shape | Purpose |
|---|---|---|
| `wte` | 597 x 32 | **Token embeddings**: one 32-dim vector per word |
| `wpe` | 16 x 32 | **Position embeddings**: one 32-dim vector per position |
| `lm_head` | 597 x 32 | **Output projection**: maps back to vocabulary |
| `layer0.attn_wq` | 32 x 32 | Attention query weights (layer 0) |
| `layer0.attn_wk` | 32 x 32 | Attention key weights (layer 0) |
| `layer0.attn_wv` | 32 x 32 | Attention value weights (layer 0) |
| `layer0.attn_wo` | 32 x 32 | Attention output weights (layer 0) |
| `layer0.mlp_fc1` | 128 x 32 | MLP hidden layer (layer 0) |
| `layer0.mlp_fc2` | 32 x 128 | MLP output layer (layer 0) |
| `layer1.attn_*` | (same) | Attention weights (layer 1) |
| `layer1.mlp_*` | (same) | MLP weights (layer 1) |

Total: **63,296 parameters**. Every one of these numbers will be adjusted
during training.

## Embeddings: How Token IDs Become Vectors

This is a crucial concept. The model cannot do math on the integer `541`
directly. It needs a *representation*, a vector of numbers that captures
something about the meaning of the word.

The **token embedding table** (`wte`) is a 597 x 32 matrix. Each row is a
32-dimensional vector representing one word in our vocabulary. To look up the
embedding for "the" (token 541):

```
wte[541] -> [0.03, -0.11, 0.07, ..., 0.02]   (32 numbers)
```

Before training, these vectors are random. The word "the" and the word "cat"
have random, meaningless embeddings. After training, words that appear in
similar contexts will have similar embeddings. The model discovers
relationships between words entirely from the data.

The **position embedding table** (`wpe`) works the same way but for positions
instead of words. `wpe[0]` is a 32-dim vector meaning "first word,"
`wpe[3]` means "fourth word." This is how the model knows word order. Without
position embeddings, it would not know whether "the cat chased the dog" and
"the dog chased the cat" are different.

The first thing the model does with any input token is combine both embeddings:

```typescript
const tokEmb = stateDict["wte"][tokenId];    // what word is this?
const posEmb = stateDict["wpe"][posId];      // where in the sentence is it?
let x = tokEmb.map((t, i) => t.add(posEmb[i]));  // combine
```

This combined vector (32 numbers encoding both the word identity and its
position) is what flows through the rest of the network.

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
|  [Multi-Head Attention]  ->  look at context  |
|  [+ Residual Connection]                      |
|  [MLP: expand -> ReLU -> compress]  -> process|
|  [+ Residual Connection]                      |
+-----------------------------------------------+
    |
+--- Transformer Layer 1 ----------------------+
|  (same structure)                             |
+-----------------------------------------------+
    |
[lm_head projection]  ->  597 raw scores (one per word)
    |
"logits", unnormalized predictions for the next word
```

The output is 597 **logits** (as introduced in Lab 04), one raw score per
word. Before training, these are essentially random. After training, if you
feed the model "the cat", the logits for "runs," "eats," and "sits" will be
much higher than "the" or "zoo."

## Inside a Transformer Layer

Each layer has two blocks, both using the primitives from Lab 04:

### Multi-Head Attention: "What Should I Pay Attention To?"

The model computes three vectors from the current input using `linear`:

- **Query (Q)**: "what am I looking for?"
- **Key (K)**: "what do I contain?"
- **Value (V)**: "what information do I provide?"

```typescript
const q = linear(x, stateDict[`layer${li}.attn_wq`]);
const k = linear(x, stateDict[`layer${li}.attn_wk`]);
const v = linear(x, stateDict[`layer${li}.attn_wv`]);
```

The attention score between the current position and each past position is the
dot product of Q and K, scaled by the square root of the head dimension. These
scores go through `softmax` to become weights, and the output is a weighted sum
of the V vectors.

This is how the model looks back at previous words in the sentence. When
processing "eats" in "the cat eats", the attention mechanism can learn to focus
on "cat" (the subject) to help predict what comes next.

The "multi-head" part means we split the 32-dim vectors into 4 heads of 8
dimensions each. Each head can attend to different things. One head might
track the subject, another might track the verb pattern.

### MLP (Feed-Forward Network): "What Do I Make of This?"

After attention gathers context, the MLP processes it:

```typescript
x = linear(x, stateDict[`layer${li}.mlp_fc1`]);  // 32 -> 128
x = x.map((xi) => xi.relu());                      // non-linearity
x = linear(x, stateDict[`layer${li}.mlp_fc2`]);  // 128 -> 32
```

The MLP expands the 32-dim vector to 128 dimensions, applies ReLU (sets
negative values to zero), and compresses back to 32. This expansion-compression
pattern gives the model room to compute complex features.

### Residual Connections: "Don't Forget"

After each block (attention and MLP), the original input is added back:

```typescript
x = x.map((a, i) => a.add(xResidual[i]));
```

This "skip connection" means the model can pass information through unchanged
if a layer has nothing useful to add. It also helps gradients flow during
training. Without residuals, deep networks are very hard to train.

## Creating the Model

The `createModel` function allocates all the weight matrices and collects
every individual number into a flat `params` array:

```typescript
export function createModel(config: GPTConfig): Model {
  const stateDict: StateDict = {
    wte: matrix(vocabSize, nEmbd),
    wpe: matrix(blockSize, nEmbd),
    lm_head: matrix(vocabSize, nEmbd),
  };

  for (let i = 0; i < nLayer; i++) {
    stateDict[`layer${i}.attn_wq`] = matrix(nEmbd, nEmbd);
    stateDict[`layer${i}.attn_wk`] = matrix(nEmbd, nEmbd);
    // ... (all attention + MLP matrices)
  }

  const params: Value[] = Object.values(stateDict)
    .flatMap((mat) => mat.flatMap((row) => row));

  return { config, stateDict, params };
}
```

The `params` array is what the optimizer will update during training. The
`stateDict` is a named view into those same parameters. When training updates
`params[i]`, the corresponding entry in `stateDict` changes too, because they
are the same `Value` objects.

## Putting It All Together

Every operation (embedding lookup, `linear` transform, `softmax`,
`rmsnorm`, addition, ReLU) is built from `Value` nodes. The entire forward
pass builds one enormous computation graph. When we call `backward()` on the
loss, the gradients for all 63,296 parameters are computed in a single sweep
through this graph.

This is what makes neural network training possible: the autograd engine turns
the question "how should I change 63,296 numbers to make my predictions
better?" into a mechanical, automatic computation.

At this point we have 63,296 random numbers and a blueprint for how to wire
them together. The model can process tokens, but its output is nonsense. To
make it useful, we need to train it.

Next: [Lab 06: Training](06-training.md)
