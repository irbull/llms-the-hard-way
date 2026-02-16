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
  encode(sentence: string): number[];
  decode(tokens: number[]): string;
}

/** Word-level tokenizer. Discovers the word vocabulary from the corpus. */
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
