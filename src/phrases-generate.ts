/**
 * MicroGPT — Generate Phrases
 *
 * Loads a trained model and generates new sentences.
 * Train first with: npx tsx phrases-train.ts
 *
 *   npx tsx phrases-generate.ts
 *   npx tsx phrases-generate.ts 50                  # generate 50 sentences
 *   npx tsx phrases-generate.ts 20 --temp=0.3       # low temperature
 *   npx tsx phrases-generate.ts 20 --top-k=10       # top-k filtering
 *   npx tsx phrases-generate.ts 20 --top-p=0.9      # nucleus sampling
 *   npx tsx phrases-generate.ts 20 --temp=0.7 --top-p=0.9 --top-k=50
 */

import { readFileSync } from "node:fs";
import { seed } from "./rng.js";
import { createWordTokenizer } from "./tokenizer.js";
import { loadModel } from "./model.js";
import { generate, type GenerateOptions } from "./generate.js";

// Parse CLI arguments
const args = process.argv.slice(2);
const positional = args.filter((a) => !a.startsWith("--"));
const flags = Object.fromEntries(
  args.filter((a) => a.startsWith("--")).map((a) => {
    const [k, v] = a.slice(2).split("=");
    return [k, v];
  })
);

const numSamples = parseInt(positional[0] ?? "20", 10);
const options: GenerateOptions = {
  temperature: parseFloat(flags["temp"] ?? "0.8"),
  topK: parseInt(flags["top-k"] ?? "0", 10),
  topP: parseFloat(flags["top-p"] ?? "1.0"),
};

// 1. Seed the RNG
seed(Date.now());

// 2. Rebuild the tokenizer from the same corpus (needed for encode/decode)
const docs = readFileSync("data/grade1_sentences.txt", "utf-8")
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => l.trim().toLowerCase());
const tokenizer = createWordTokenizer(docs);

// 3. Load the trained model
const model = loadModel("phrases-model.json");
console.log(`loaded model: ${model.params.length} params, vocab ${model.config.vocabSize}`);

// 4. Generate
const parts = [`temperature=${options.temperature}`];
if (options.topK! > 0) parts.push(`top-k=${options.topK}`);
if (options.topP! < 1.0) parts.push(`top-p=${options.topP}`);
console.log(`generating ${numSamples} sentences (${parts.join(", ")}):\n`);

const sentences = generate(model, tokenizer, numSamples, options);
sentences.forEach((s, i) =>
  console.log(`  ${String(i + 1).padStart(2)}. ${s}`)
);
