/**
 * Inference / Text Generation
 *
 * Starting from the BOS token, the model predicts one token at a time:
 * 1. Forward the current token through the model to get logits
 * 2. Apply temperature scaling (lower = more conservative, higher = more creative)
 * 3. Apply top-k filtering (keep only the k most likely tokens)
 * 4. Apply top-p / nucleus filtering (keep smallest set summing to p)
 * 5. Convert to probabilities via softmax
 * 6. Randomly sample the next token from that distribution
 * 7. Stop when BOS is produced (end of sequence) or max length is reached
 */

import { softmax } from "./nn.js";
import { type Model, gpt, createKVCache } from "./model.js";
import { weightedChoice } from "./rng.js";
import type { Tokenizer } from "./tokenizer.js";

export interface GenerateOptions {
  temperature?: number;
  topK?: number;
  topP?: number;
}

/** Generate new text samples from a trained model. */
export function generate(
  model: Model,
  tokenizer: Tokenizer,
  numSamples: number,
  options: GenerateOptions = {},
): string[] {
  const { temperature = 0.8, topK = 0, topP = 1.0 } = options;
  const samples: string[] = [];

  for (let i = 0; i < numSamples; i++) {
    const { keys, values } = createKVCache(model);
    let tokenId = tokenizer.BOS;
    const tokens: number[] = [];

    for (let posId = 0; posId < model.config.blockSize; posId++) {
      const logits = gpt(model, tokenId, posId, keys, values);

      // Temperature scaling
      let scores = logits.map((l) => l.data / temperature);

      // Top-k: keep only the k highest scores
      if (topK > 0 && topK < scores.length) {
        const sorted = [...scores].sort((a, b) => b - a);
        const cutoff = sorted[topK];
        scores = scores.map((s) => (s >= cutoff ? s : -Infinity));
      }

      // Top-p (nucleus): keep smallest set whose probabilities sum to p
      if (topP < 1.0) {
        const indices = scores.map((s, idx) => idx);
        indices.sort((a, b) => scores[b] - scores[a]);
        // Compute softmax on current scores to get probabilities for filtering
        const maxS = Math.max(...scores.filter((s) => s !== -Infinity));
        const exps = scores.map((s) => (s === -Infinity ? 0 : Math.exp(s - maxS)));
        const total = exps.reduce((a, b) => a + b, 0);
        const probs = exps.map((e) => e / total);
        let cumSum = 0;
        for (const idx of indices) {
          cumSum += probs[idx];
          if (cumSum > topP) {
            scores[idx] = -Infinity;
          }
        }
      }

      // Softmax and sample (using plain numbers, not Value, for efficiency)
      const maxS = Math.max(...scores.filter((s) => s !== -Infinity));
      const exps = scores.map((s) => (s === -Infinity ? 0 : Math.exp(s - maxS)));
      const total = exps.reduce((a, b) => a + b, 0);
      const probs = exps.map((e) => e / total);

      tokenId = weightedChoice(probs);
      if (tokenId === tokenizer.BOS) break;
      tokens.push(tokenId);
    }

    samples.push(tokenizer.decode(tokens));
  }

  return samples;
}
