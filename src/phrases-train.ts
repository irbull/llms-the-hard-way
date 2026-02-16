/**
 * MicroGPT — Train on Phrases
 *
 * Trains a word-level GPT on grade 1 English sentences and saves the model.
 * Run inference afterwards with: npx tsx phrases-generate.ts
 *
 *   npx tsx phrases-train.ts
 */

import { readFileSync } from "node:fs";
import { seed, shuffle } from "./rng.js";
import { createWordTokenizer } from "./tokenizer.js";
import { createModel, saveModel } from "./model.js";
import { train } from "./train.js";

// 1. Seed the RNG for reproducibility
seed(42);

// 2. Load dataset: 30K grade 1 sentences
const docs = readFileSync("data/grade1_sentences.txt", "utf-8")
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => l.trim().toLowerCase());
shuffle(docs);
console.log(`num sentences: ${docs.length}`);

// 3. Build word-level tokenizer from the corpus
const tokenizer = createWordTokenizer(docs);
console.log(`vocab size: ${tokenizer.vocabSize} (${tokenizer.vocabSize - 1} words + BOS)`);

// 4. Create the model — bigger than the names model to handle the larger vocabulary
const untrained = createModel({
  nLayer: 2,
  nEmbd: 32,
  blockSize: 16,
  nHead: 4,
  headDim: 8,
  vocabSize: tokenizer.vocabSize,
});
console.log(`num params: ${untrained.params.length}`);

// 5. Train the model
const model = train(
  untrained,
  { numSteps: 5000, learningRate: 0.01, beta1: 0.85, beta2: 0.99, epsAdam: 1e-8 },
  docs,
  tokenizer,
);

// 6. Save the trained model to disk
saveModel(model, "phrases-model.json");
console.log("\nmodel saved to phrases-model.json");
