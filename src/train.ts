/**
 * Training Loop
 *
 * Each step:
 * 1. Pick a document and tokenize it
 * 2. Forward each token through the model, predicting the next token
 * 3. Compute cross-entropy loss (how surprised the model is at the correct answer)
 * 4. Backward pass: compute gradients via the chain rule
 * 5. Adam optimizer: update parameters to reduce loss
 *
 * Loss starts at ~3.3 (random guessing among 27 tokens = -log(1/27))
 * and decreases to ~2.37 over 1000 steps.
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
  docs: string[],
  tokenizer: Tokenizer,
): Model {
  const { numSteps, learningRate, beta1, beta2, epsAdam } = trainConfig;
  const { params } = model;

  // Adam optimizer moment buffers
  const mBuf = new Float64Array(params.length);
  const vBuf = new Float64Array(params.length);

  for (let step = 0; step < numSteps; step++) {
    // Tokenize a single document, surrounded by BOS on both sides
    const doc = docs[step % docs.length];
    const tokens = tokenizer.encode(doc);
    const n = Math.min(model.config.blockSize, tokens.length - 1);

    // Forward: build the computation graph from tokens to loss
    const { keys, values } = createKVCache(model);
    const losses: Value[] = [];

    for (let posId = 0; posId < n; posId++) {
      const tokenId = tokens[posId];
      const targetId = tokens[posId + 1];
      const logits = gpt(model, tokenId, posId, keys, values);
      const probs = softmax(logits);
      losses.push(probs[targetId].log().neg());
    }

    // Average cross-entropy loss over the sequence
    const loss = vsum(losses).div(n);

    // Backward: compute gradients for all parameters
    loss.backward();

    // Adam update with linear learning rate decay
    const lrT = learningRate * (1 - step / numSteps);
    for (let i = 0; i < params.length; i++) {
      mBuf[i] = beta1 * mBuf[i] + (1 - beta1) * params[i].grad;
      vBuf[i] = beta2 * vBuf[i] + (1 - beta2) * params[i].grad ** 2;
      const mHat = mBuf[i] / (1 - beta1 ** (step + 1));
      const vHat = vBuf[i] / (1 - beta2 ** (step + 1));
      params[i].data -= lrT * mHat / (Math.sqrt(vHat) + epsAdam);
      params[i].grad = 0;
    }

    process.stdout.write(
      `\rstep ${String(step + 1).padStart(4)} / ${numSteps} | loss ${loss.data.toFixed(4)}`
    );
  }

  return model;
}
