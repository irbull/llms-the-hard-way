# Encoding and Decoding

## Encoding: Text to Numbers

To encode a sentence, we wrap it in BOS tokens and replace each word with its
index:

```
"the cat eats a muffin"
   | encode
[596, 525, 75, 152, 0, 324, 596]
 BOS  the  cat eats a  muffin BOS
```

The BOS token appears at both ends. The opening BOS gives the model a
consistent starting signal. The closing BOS tells the model the sentence is
complete. During training, the model learns that after certain patterns of
words, the next token should be BOS, meaning "stop."

## Decoding: Numbers Back to Text

Decoding reverses the process. Given `[525, 75, 152, 0, 324]`, we look up
each index in the vocabulary and join with spaces:

```
[525, 75, 152, 0, 324]
  | decode
"the cat eats a muffin"
```

## The Code

The tokenizer's API is captured in a `Tokenizer` interface:

```typescript
// tokenizer.ts
export interface Tokenizer {
  vocabSize: number;
  BOS: number;
  encode(sentence: string): number[];
  decode(tokens: number[]): string;
}
```

A factory function builds one from the training corpus:

```typescript
export function createWordTokenizer(sentences: string[]): Tokenizer {
  const words = [...new Set(sentences.flatMap((d) => d.split(" ")))].sort();
  const BOS = words.length;
  const vocabSize = words.length + 1;

  return {
    vocabSize,
    BOS,
    encode(sentence: string): number[] {
      return [BOS, ...sentence.split(" ").map((w) => words.indexOf(w)), BOS];
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
