/**
 * MicroGPT — Fine-Tune on Questions
 *
 * Loads a pre-trained model and fine-tunes it on a small question dataset.
 * The model learns to produce questions while preserving its existing knowledge.
 *
 *   npx tsx src/phrases-fine-tune.ts
 */

import { readFileSync } from "node:fs";
import { seed, shuffle } from "./rng.js";
import { createWordTokenizer } from "./tokenizer.js";
import { loadModel, saveModel } from "./model.js";
import { train } from "./train.js";

// 1. Seed the RNG for reproducibility
seed(42);

// 2. Load the ORIGINAL corpus to build the SAME tokenizer.
//    The tokenizer must match the one used during pre-training,
//    because the model's embeddings are tied to specific token IDs.
const originalDocs = readFileSync("data/grade1_sentences.txt", "utf-8")
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => l.trim().toLowerCase());
const tokenizer = createWordTokenizer(originalDocs);
console.log(`vocab size: ${tokenizer.vocabSize} (${tokenizer.vocabSize - 1} words + BOS)`);

// 3. Load the pre-trained model
const model = loadModel("phrases-model.json");
console.log(`loaded pre-trained model: ${model.params.length} params`);

// 4. Load the fine-tuning dataset (questions)
const questionDocs = readFileSync("data/questions.txt", "utf-8")
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => l.trim().toLowerCase());
shuffle(questionDocs);
console.log(`fine-tuning sentences: ${questionDocs.length}`);

// 5. Fine-tune: fewer steps, lower learning rate.
//    - Lower learning rate (0.001 vs 0.01) preserves existing knowledge
//    - Fewer steps (1000 vs 5000) since we're adapting, not learning from scratch
const fineTuned = train(
  model,
  { numSteps: 1000, learningRate: 0.001, beta1: 0.85, beta2: 0.99, epsAdam: 1e-8 },
  questionDocs,
  tokenizer,
);

// 6. Save the fine-tuned model
saveModel(fineTuned, "phrases-fine-tuned-model.json");
console.log("\nmodel saved to phrases-fine-tuned-model.json");
