# Multi-Head Attention

Each transformer layer has two blocks, both using the primitives from the previous chapter.
The first is multi-head attention — the mechanism that lets the model look at
previous tokens to inform its prediction.

## Three Identical Projections

The model computes three vectors from the current hidden state using `linear`:

```typescript
const query: Value[] = linear(hidden, layer.attention.query);
const key: Value[]   = linear(hidden, layer.attention.key);
const value: Value[] = linear(hidden, layer.attention.value);
```

These are three matrix multiplications with three different weight matrices, all
the same shape (`[32 × 32]`). Before training, they contain random numbers.
There is nothing in this code that makes `query` behave differently from `key`
or `value` — they are structurally identical linear projections of the same
hidden state.

The names describe what these projections *become*, not what they *are*. The
labels Q, K, and V are aspirational: they describe roles that emerge through
training, not roles assigned by the code. What forces them into distinct roles
is the math that follows.

## The Attention Computation

Here is what happens to Q, K, and V once they exist:

```
attention = softmax(Q · K^T / √headDim) × V
```

This structure imposes hard constraints:

- **Q and K always get dot-producted** to produce a similarity score. They are
  structurally forced into a "matching" role — whatever W_q and W_k produce
  will be compared against each other.
- **V always gets weighted-summed** using those scores. It is structurally
  forced into a "what gets retrieved" role — the scores from Q·K gate how
  much of each V vector flows through.

What's *learned* is everything else: what Q and K match *on* (syntactic role?
position? something we can't name?), and what V carries (whatever information
turned out to be useful to retrieve). Q/K/V really are like query/key/value in
a database lookup — but the schema of that database, what counts as a match,
what's worth storing — all of that is discovered by training.

The `√headDim` scaling prevents the dot products from growing so large that
softmax saturates into near-zero gradients, which would kill learning.

Here is how the math notation maps to our code:

| Math | Code | Shape |
|------|------|-------|
| Q | `query` (or `headQuery` per head) | 32-dim (8 per head) |
| K | `key` (or `headKeys` per head) | 32-dim (8 per head) |
| V | `value` (or `headValues` per head) | 32-dim (8 per head) |
| W_q, W_k, W_v | `layer.attention.query`, `.key`, `.value` | [32 × 32] |
| √d_k | `Math.pow(headDim, -0.5)` | scalar (√8) |

## Worked Example: "the cat eats"

Suppose we're processing the sentence "BOS the cat eats" and we're on "eats"
at position 3, trying to predict what comes next. The 32-dim hidden state for
"eats" gets projected into Q, K, and V. Each head works on its own 8-dim
slice, so let's follow one head.

```
Q_eats = hidden @ W_q   →  an 8-dim vector for this head
```

The KV cache already holds the key and value vectors for every previous
position (more on the cache later). So we have:

```
K_BOS, K_the, K_cat, K_eats     (cached keys for positions 0, 1, 2, 3)
V_BOS, V_the, V_cat, V_eats     (cached values for positions 0, 1, 2, 3)
```

Compute attention scores — how much should "eats" attend to each position:

```
score("eats", BOS)    = Q_eats · K_BOS  / √8 = 0.1    (low)
score("eats", "the")  = Q_eats · K_the  / √8 = 0.5    (low)
score("eats", "cat")  = Q_eats · K_cat  / √8 = 2.5    (high)
score("eats", "eats") = Q_eats · K_eats / √8 = 1.0    (medium)
```

After softmax, these become weights that sum to 1.0. The output is a weighted
sum of the value vectors:

```
output = 0.02 × V_BOS  +  0.08 × V_the  +  0.66 × V_cat  +  0.24 × V_eats
```

"cat" dominates — the information encoded in V_cat flows through strongly.
This is how the model routes information from relevant earlier tokens to the
current position.

To be honest: those scores are made up. What actually causes Q_eats · K_cat
to be high? The weight matrices W_q and W_k learned — from millions of
training examples — to project words into a space where verbs and their
subjects end up pointing in similar directions. We can observe that this
happens. We can't easily read off *why* from the weight matrices.

## Causal Masking

Our model is autoregressive — it predicts the next token from previous ones.
Each position can only attend to itself and earlier positions, never future
ones. In our implementation this happens naturally: the KV cache only contains
entries for tokens that have already been processed, so there are no future
keys to attend to.

## Multiple Heads

The "multi-head" part means we split the 32-dim vectors into 4 heads of 8
dimensions each. Each head has its own slice of W_q, W_k, and W_v, so each
head runs its own independent attention computation in parallel:

```typescript
for (let h = 0; h < nHead; h++) {
  const headStart = h * headDim;
  const headQuery  = query.slice(headStart, headStart + headDim);   // this head's 8-dim query
  const headKeys   = keys[li].map((ki) => ki.slice(headStart, headStart + headDim));
  const headValues = values[li].map((vi) => vi.slice(headStart, headStart + headDim));
  // compute attention scores and weighted sum for this head
}
```

Different heads learn to attend to different things. On "the cat eats", the
four heads might specialize like this:

```
Head 0 — tracks grammatical subject:
  score("eats" → "cat") = 0.9    →  output mostly carries V_cat

Head 1 — tracks recency / local context:
  score("eats" → "eats") = 0.8   →  output mostly carries V_eats

Head 2 — tracks determiner–noun relationships:
  score("cat" → "the") = 0.9     →  output mostly carries V_the

Head 3 — honestly, nobody knows:
  output = some mix that turns out to be useful
```

The outputs from all four heads are concatenated back into a 32-dim vector
and projected through one more linear layer (`layer.attention.output`) to
produce the final attention output.

By the time we're done, the representation of "eats" has been enriched by: who
the subject is, where we are in the sequence, what the determiner structure
looks like, and whatever Head 3 noticed.

## Why Heads Don't All Learn the Same Thing

Each head starts with different random weights. Gradient descent is local — it
nudges each head's weights based on where they started, so they drift in
different directions. And there's no benefit to convergence: redundant heads
contribute identical information, giving the model no extra expressive power.
The loss landscape rewards diversity because diverse heads capture more signal.

One consequence: the *capabilities* are reproducible but their *locations* are
not. Train the same model twice with different random seeds and you'll get
subject-verb tracking somewhere, but not necessarily in the same head or layer.
The capability emerges; its address doesn't.
