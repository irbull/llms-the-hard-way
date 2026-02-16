# Running the Model

The `gpt()` function wires together every piece from this chapter: it takes a
single token and its position, runs it through the full architecture, and
returns 597 logits. Let's walk through it before looking at the code.

## Step by Step

The function takes five arguments: the model, a token ID, a position, and the
KV cache (keys and values) introduced earlier in this chapter.

**1. Embedding lookup.** Look up the token embedding and the position
embedding, and add them together to produce a single 32-element vector:

```typescript
const tokenVector: Value[] = weights.tokenEmbedding[tokenId];
const positionVector: Value[] = weights.positionEmbedding[posId];
let hidden: Value[] = tokenVector.map((t, i) => t.add(positionVector[i]));
```

At this point `hidden` is a 32-dim vector that encodes both *which word* this
is and *where* it appears in the sentence.

**2. Normalize.** Apply rmsnorm to keep the numbers at a stable scale:

```typescript
hidden = rmsnorm(hidden);
```

**3. Transformer layers.** The model has 2 layers. Each one does the same
thing: attention, then MLP, each with a residual connection.

**3a. Attention.** Save the hidden state for the residual connection, normalize,
then compute query, key, and value vectors. Notice that normalization comes
*before* each block, not after — this is called **pre-norm** and is what
GPT-2 and LLaMA use. The original transformer paper normalized *after* each
block (post-norm), but pre-norm trains more stably because the residual
connection carries unnormalized values, giving gradients a cleaner path
backward. Push the key and value into the KV
cache so future tokens can attend to this position:

```typescript
const beforeAttention: Value[] = hidden;
hidden = rmsnorm(hidden);

const query: Value[] = linear(hidden, layer.attention.query);
const key: Value[]   = linear(hidden, layer.attention.key);
const value: Value[] = linear(hidden, layer.attention.value);
keys[li].push(key);
values[li].push(value);
```

Then split into heads (4 heads of 8 dimensions each). For each head, compute
attention scores by comparing this token's query against all cached keys, run
softmax to get weights, and take a weighted sum of the cached values. This is
how the model looks at previous tokens to gather context.

Finally, project the concatenated head outputs back to 32 dimensions and add
the residual:

```typescript
hidden = linear(attentionOutput, layer.attention.output);
hidden = hidden.map((h, i) => h.add(beforeAttention[i]));
```

**3b. MLP.** Save the hidden state for the residual, normalize, expand from 32
to 128 dimensions, apply ReLU, compress back to 32, and add the residual:

```typescript
const beforeMLP: Value[] = hidden;
hidden = rmsnorm(hidden);
hidden = linear(hidden, layer.mlp.hidden);     // 32 -> 128
hidden = hidden.map((h) => h.relu());          // non-linearity
hidden = linear(hidden, layer.mlp.output);     // 128 -> 32
hidden = hidden.map((h, i) => h.add(beforeMLP[i]));
```

**4. Output projection.** After all layers, project the 32-dim vector to 597
dimensions — one score per word in the vocabulary:

```typescript
return linear(hidden, weights.output);
```

These 597 numbers are the **logits**: the model's raw, unnormalized prediction
for what word comes next.

## The Complete Function

Here is the full `gpt()` function with all the pieces together:

```typescript
export function gpt(
  model: Model,
  tokenId: number,
  posId: number,
  keys: Value[][][],
  values: Value[][][],
): Value[] {
  const { nLayer, nHead, headDim } = model.config;
  const { weights } = model;

  // Step 1: Embedding lookup
  // Combine "what word is this?" with "where does it appear?" into a single
  // vector. This is the hidden state that flows through the rest of the network.
  const tokenVector: Value[] = weights.tokenEmbedding[tokenId];
  const positionVector: Value[] = weights.positionEmbedding[posId];
  let hidden: Value[] = tokenVector.map((t, i) => t.add(positionVector[i]));

  // Normalize before the first layer to keep values at a stable scale
  hidden = rmsnorm(hidden);

  // Step 2: Transformer layers
  // Each layer has two blocks: attention (gather context from other tokens)
  // followed by MLP (process the gathered information). Both use residual
  // connections so the input is added back to the output of each block.
  for (let li = 0; li < nLayer; li++) {
    const layer = weights.layers[li];

    // --- Attention block: look at previous tokens to gather context ---

    // Save the hidden state so we can add it back after the block (residual)
    const beforeAttention: Value[] = hidden;
    hidden = rmsnorm(hidden);

    // Project the hidden state into query, key, and value vectors.
    // These are structurally identical projections — training teaches them
    // to play different roles: query asks "what am I looking for?",
    // key advertises "what do I contain?", value carries "what to retrieve".
    const query: Value[] = linear(hidden, layer.attention.query);
    const key: Value[]   = linear(hidden, layer.attention.key);
    const value: Value[] = linear(hidden, layer.attention.value);

    // Cache the key and value so future tokens can attend to this position
    keys[li].push(key);
    values[li].push(value);

    // Each head independently attends to a different slice of the vectors,
    // allowing the model to track multiple relationships at once
    const attentionOutput: Value[] = [];
    for (let h = 0; h < nHead; h++) {
      const headStart = h * headDim;
      const headQuery = query.slice(headStart, headStart + headDim);
      const headKeys = keys[li].map((ki) => ki.slice(headStart, headStart + headDim));
      const headValues = values[li].map((vi) => vi.slice(headStart, headStart + headDim));

      // Scaled dot-product attention: score = (query · key) / √headDim
      const attnLogits = headKeys.map((cachedKey) =>
        vsum(headQuery.map((q, j) => q.mul(cachedKey[j]))).div(Math.sqrt(headDim))
      );

      // Softmax converts scores into weights that sum to 1
      const attnWeights = softmax(attnLogits);

      // Weighted sum of value vectors — high-scoring positions contribute more
      for (let j = 0; j < headDim; j++) {
        attentionOutput.push(vsum(attnWeights.map((w, t) => w.mul(headValues[t][j]))));
      }
    }

    // Project concatenated head outputs back to the hidden dimension
    hidden = linear(attentionOutput, layer.attention.output);
    // Residual connection: add back what we had before attention
    hidden = hidden.map((h, i) => h.add(beforeAttention[i]));

    // --- MLP block: process each token's representation independently ---

    const beforeMLP: Value[] = hidden;
    hidden = rmsnorm(hidden);
    hidden = linear(hidden, layer.mlp.hidden);   // expand to 4x wider
    hidden = hidden.map((h) => h.relu());         // nonlinearity
    hidden = linear(hidden, layer.mlp.output);    // compress back
    // Residual connection: add back what we had before the MLP
    hidden = hidden.map((h, i) => h.add(beforeMLP[i]));
  }

  // Step 3: Output projection
  // Project the final hidden state to vocabulary size — one score per word.
  // These raw scores (logits) will be passed through softmax later to get
  // a probability distribution over the next token.
  // Note: standard GPT-2 and LLaMA apply a final normalization here
  // (rmsnorm(hidden) before the linear). We skip it for simplicity.
  return linear(hidden, weights.output);
}
```

Every operation here uses `Value` nodes, so the entire forward pass builds a
computation graph. When we call `backward()` on the loss later, gradients flow
back through every step — from the output projection, through the MLP and
attention blocks, all the way to the embedding lookup.
