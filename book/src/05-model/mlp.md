# MLP (Feed-Forward Network)

Each transformer layer has two blocks. Attention was the first: it mixes
information *between* tokens. The MLP is the second, and it does something
complementary: it processes each token's vector *independently*.

That distinction matters. Attention is how "cat" gets routed into the
representation of "eats." But attention is mostly a weighted average; it
selects and combines, it doesn't do much computation on its own. The MLP is
where per-token nonlinear computation happens. It transforms each vector
through learned functions that training has shaped to be useful.

Attention moves information around. The MLP does something with it.

## The Code

```typescript
hidden = linear(hidden, layer.mlp.hidden);     // 32 -> 128
hidden = hidden.map((h) => h.relu());          // non-linearity
hidden = linear(hidden, layer.mlp.output);     // 128 -> 32
```

Three operations: expand, apply nonlinearity, compress.

## Why Expand and Compress?

The first linear layer projects the 32-dim vector into 128 dimensions, a 4x
expansion. The second projects it back down to 32. Why not just do a single
32→32 transformation?

The expanded 128-dim space gives the model room to compute. In 32 dimensions,
each element has to carry the final representation directly. In 128 dimensions,
the model can build intermediate features (combinations and interactions) that
are useful for computing the final output but don't need to survive in the
compressed result.

Think of it as scratch space. The expansion gives 128 dimensions to work in;
the compression selects what's worth keeping.

## Why ReLU Is Critical

ReLU sets negative values to zero: `relu(x) = max(0, x)`. This is the
nonlinearity between the two linear layers, and without it, the MLP would be
pointless.

Here's why: a linear transformation followed by another linear transformation
is just... a single linear transformation. Matrix multiplication composes.
`W₂(W₁x) = (W₂W₁)x`. You could replace the two weight matrices with one
matrix that does the same thing. No matter how many linear layers you stack,
they collapse into one.

ReLU breaks this. By zeroing out different elements depending on the input, it
makes the transformation *input-dependent*: different inputs activate different
subsets of the 128 hidden neurons. This is what lets the MLP compute functions
that aren't just weighted sums. It's what makes depth useful.

## What Does It Compute?

Same honesty as attention: we can observe what the MLP does in aggregate, but
the individual weight values aren't interpretable. After attention has gathered
context into the "eats" vector, enriching it with information about the subject,
position, and determiner structure, the MLP applies a learned nonlinear
transformation to that enriched representation.

What patterns does it recognize? What features does it compute? Training shaped
the weights to reduce prediction error. The results work. The internals are
opaque.
