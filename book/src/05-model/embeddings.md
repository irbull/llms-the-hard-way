# Embeddings: How Token IDs Become Vectors

This is a crucial concept. The model cannot do math on the integer `541`
directly. It needs a *representation*, a vector of numbers that captures
something about the meaning of the word.

The **token embedding table** (`tokenEmbedding`) is a 597 x 32 matrix. Each row is a
32-dimensional vector representing one word in our vocabulary. To look up the
embedding for "the" (token 541):

```
tokenEmbedding[541] -> [0.03, -0.11, 0.07, ..., 0.02]   (32 numbers)
```

Before training, these vectors are random. The word "the" and the word "cat"
have random, meaningless embeddings. After training, words that appear in
similar contexts will have similar embeddings. The model discovers
relationships between words entirely from the data.

The **position embedding table** (`positionEmbedding`) works the same way but for positions
instead of words. `positionEmbedding[0]` is a 32-dim vector meaning "first word,"
`positionEmbedding[3]` means "fourth word." This is how the model knows word order. Without
position embeddings, it would not know whether "the cat chased the dog" and
"the dog chased the cat" are different.

The first thing the model does with any input token is combine both embeddings:

```typescript
const tokenVector: Value[] = weights.tokenEmbedding[tokenId];    // what word is this?
const positionVector: Value[] = weights.positionEmbedding[posId]; // where in the sentence is it?
let hidden: Value[] = tokenVector.map((t, i) => t.add(positionVector[i])); // combine
```

This combined vector, `hidden`, is 32 numbers encoding both the word identity
and its position. It is the hidden state that flows through the rest of the
network.

Notice that embedding lookup is **array indexing**, not a matrix multiply.
When the model processes token 541, only row 541 of the embedding table enters
the computation graph. The other 596 rows are never touched; they receive no
gradients during the backward pass and are not updated. Over many training
sentences different words appear, so different rows get updated at different
times. Words that appear frequently develop better embeddings than rare ones,
simply because they get more gradient updates.
