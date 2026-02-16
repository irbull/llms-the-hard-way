/**
 * Tokenizer
 *
 * Translates strings to sequences of integers ("tokens") and back.
 * Builds a word-level vocabulary from the training corpus, with a BOS
 * (Beginning of Sequence) delimiter appended on each side.
 */

export interface Tokenizer {
  vocabSize: number;
  BOS: number;
  encode(doc: string): number[];
  decode(tokens: number[]): string;
}

/** Word-level tokenizer. Discovers the word vocabulary from the corpus. */
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
