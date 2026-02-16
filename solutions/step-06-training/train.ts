/**
 * Training Loop
 *
 * Each step processes one sentence through five stages:
 * 1. Pick a sentence and tokenize it
 * 2. Forward: feed tokens one at a time, predicting the next token at each position
 * 3. Compute cross-entropy loss (how surprised the model is at the correct answer)
 * 4. Backward: compute gradients via the chain rule
 * 5. Adam optimizer: update parameters to reduce loss
 *
 * Loss starts at ~6.39 (random guessing among 597 tokens = -log(1/597))
 * and decreases to ~1.76 over 5,000 steps.
 */

import { Value, vsum } from "./autograd.js";
import { softmax } from "./nn.js";
import { type Model, gpt, createKVCache } from "./model.js";
import type { Tokenizer } from "./tokenizer.js";

export interface TrainConfig {
  numSteps: number;
  learningRate: number;
  beta1: number;
  beta2: number;
  epsAdam: number;
}

/** Train the model on the dataset and return the trained model. */
export function train(
  model: Model,
  trainConfig: TrainConfig,
  sentences: string[],
  tokenizer: Tokenizer,
): Model {
  const { numSteps, learningRate, beta1, beta2, epsAdam } = trainConfig;
  const { params } = model;

  // Adam optimizer maintains two running averages per parameter:
  // mBuf: smoothed gradient direction (momentum)
  // vBuf: smoothed gradient magnitude (for adaptive step sizes)
  const mBuf = new Float64Array(params.length);
  const vBuf = new Float64Array(params.length);

  for (let step = 0; step < numSteps; step++) {
    // --- Step 1: Pick a sentence and tokenize ---
    // Cycle through training data; the tokenizer wraps the sentence with BOS markers
    const sentence = sentences[step % sentences.length];
    const tokens = tokenizer.encode(sentence);
    const n = Math.min(model.config.blockSize, tokens.length - 1);

    // --- Step 2: Forward pass ---
    // Feed tokens one at a time, building a computation graph.
    // Each gpt() call adds to the KV cache so later tokens can attend to earlier ones.
    const { keys, values } = createKVCache(model);
    const losses: Value[] = [];

    for (let posId = 0; posId < n; posId++) {
      const tokenId = tokens[posId];
      const targetId = tokens[posId + 1];  // the word we want the model to predict
      const logits = gpt(model, tokenId, posId, keys, values);
      const probs = softmax(logits);
      // Cross-entropy loss: -log(probability assigned to the correct word)
      // High probability on the right word → low loss; low probability → high loss
      losses.push(probs[targetId].log().neg());
    }

    // Average loss over the sentence so short and long sentences contribute equally
    const loss = vsum(losses).div(n);

    // --- Step 3: Backward pass ---
    // Walk the computation graph and fill in .grad on every Value node.
    // After this, each parameter knows how much it contributed to the loss.
    loss.backward();

    // --- Step 4: Adam optimizer with linear learning rate decay ---
    // Large steps early (lr = 0.01) to learn quickly, decaying to 0 to fine-tune
    const lrT = learningRate * (1 - step / numSteps);
    for (let i = 0; i < params.length; i++) {
      // Update momentum (smoothed gradient direction)
      mBuf[i] = beta1 * mBuf[i] + (1 - beta1) * params[i].grad;
      // Update velocity (smoothed gradient magnitude)
      vBuf[i] = beta2 * vBuf[i] + (1 - beta2) * params[i].grad ** 2;
      // Bias correction: counteract the zero-initialization of mBuf/vBuf
      const mHat = mBuf[i] / (1 - beta1 ** (step + 1));
      const vHat = vBuf[i] / (1 - beta2 ** (step + 1));
      // Update: step in the smoothed gradient direction, scaled by the inverse
      // of the gradient magnitude so sensitive parameters take smaller steps
      params[i].data -= lrT * mHat / (Math.sqrt(vHat) + epsAdam);
      params[i].grad = 0;  // reset for next sentence
    }

    process.stdout.write(
      `\rstep ${String(step + 1).padStart(4)} / ${numSteps} | loss ${loss.data.toFixed(4)}`
    );
  }

  return model;
}
