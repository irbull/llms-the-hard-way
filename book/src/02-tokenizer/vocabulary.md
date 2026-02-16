# Building the Vocabulary

We scan every sentence in the training data and collect every unique word:

```typescript
const words = [...new Set(sentences.flatMap((d) => d.split(" ")))].sort();
```

This gives us a sorted list of 596 words:

```
Index 0:   "a"
Index 1:   "all"
Index 2:   "am"
...
Index 75:  "cat"
Index 76:  "catch"
...
Index 525: "the"
Index 526: "then"
...
Index 595: "zoo"
```

We then add one special token: **BOS** (Beginning of Sequence), assigned index
596. This marker tells the model "a sentence starts here" and "a sentence ends
here." Production models typically use separate BOS and EOS (End of Sequence)
tokens. We use a single token for both roles — the model sees BOS at the start
and learns that predicting BOS means the sentence is over.

Total vocabulary size: **597** (596 words + 1 BOS token).
