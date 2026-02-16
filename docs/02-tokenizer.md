# Lab 02: The Tokenizer

Neural networks operate on numbers, not words. The tokenizer is the bridge: it
assigns every word in our vocabulary a unique integer, then translates
sentences back and forth between text and number sequences.

## Building the Vocabulary

We scan every sentence in the training data and collect every unique word:

```typescript
const words = [...new Set(docs.flatMap((d) => d.split(" ")))].sort();
```

This gives us a sorted list of 596 words:

```
Index 0:   "a"
Index 1:   "about"
Index 2:   "across"
...
Index 89:  "cat"
Index 90:  "catch"
...
Index 541: "the"
Index 542: "them"
...
Index 595: "zoo"
```

We then add one special token: **BOS** (Beginning of Sequence), assigned index
596. This marker tells the model "a sentence starts here" and "a sentence ends
here."

Total vocabulary size: **597** (596 words + 1 BOS token).

## Encoding: Text to Numbers

To encode a sentence, we wrap it in BOS tokens and replace each word with its
index:

```
"the cat eats a muffin"
   | encode
[596, 541, 89, 152, 0, 358, 596]
 BOS  the  cat eats a  muffin BOS
```

The BOS token appears at both ends. The opening BOS gives the model a
consistent starting signal. The closing BOS tells the model the sentence is
complete. During training, the model learns that after certain patterns of
words, the next token should be BOS, meaning "stop."

## Decoding: Numbers Back to Text

Decoding reverses the process. Given `[541, 89, 152, 0, 358]`, we look up
each index in the vocabulary and join with spaces:

```
[541, 89, 152, 0, 358]
  | decode
"the cat eats a muffin"
```

## Why Word-Level Tokens?

An alternative approach is character-level tokenization, where each letter is
a token. Our vocabulary would be just 27 tokens (a-z plus BOS), but the
sequences become much longer: "the cat" becomes 7 character tokens instead of
2 word tokens. With only 596 unique words in our corpus, word-level tokenization
gives us short sequences that are fast to train on and easy to reason about.

Production LLMs like GPT-4 use a middle ground called [**subword tokenization**
(BPE)](https://medium.com/data-science/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0), which splits rare words into pieces while keeping common words whole.
Our word-level approach is the simplest version of this idea.

## The Code

The tokenizer is a factory function that returns an object with `encode` and
`decode`:

```typescript
// tokenizer.ts
export function createWordTokenizer(docs: string[]): Tokenizer {
  const words = [...new Set(docs.flatMap((d) => d.split(" ")))].sort();
  const BOS = words.length;
  const vocabSize = words.length + 1;

  return {
    vocabSize,
    BOS,
    encode(doc: string): number[] {
      return [BOS, ...doc.split(" ").map((w) => words.indexOf(w)), BOS];
    },
    decode(tokens: number[]): string {
      return tokens.map((t) => words[t]).join(" ");
    },
  };
}
```

The tokenizer has no knowledge of English. It does not know that "the" is an
article or that "cat" is a noun. It just maps strings to integers. All the
meaning will come from training.

Next: [Lab 03: The Autograd Engine](03-autograd.md)
