/**
 * Neural Network Primitives
 *
 * The building blocks that the GPT architecture assembles:
 * - linear: matrix-vector multiply (the fundamental neural net operation)
 * - softmax: convert raw scores into a probability distribution
 * - rmsnorm: normalize activations to stabilize training
 *
 * These mirror PyTorch's torch.nn.functional — general-purpose operations
 * used by the model, training loop, and inference.
 */

import { Value, vsum } from "./autograd.js";

export type Matrix = Value[][];

/** y = Wx: multiply a weight matrix by an input vector. */
export function linear(input: Value[], weights: Matrix): Value[] {
  return weights.map((row) => vsum(row.map((w, i) => w.mul(input[i]))));
}

/** Convert raw logits to probabilities. Subtracts max for numerical stability. */
export function softmax(logits: Value[]): Value[] {
  const maxVal = Math.max(...logits.map((v) => v.data));
  const exps = logits.map((v) => v.sub(maxVal).exp());
  const total = vsum(exps);
  return exps.map((e) => e.div(total));
}

/** Root Mean Square normalization: scale activations to unit variance. */
export function rmsnorm(input: Value[]): Value[] {
  const ms = vsum(input.map((xi) => xi.mul(xi))).div(input.length);
  const scale = ms.add(1e-5).pow(-0.5);
  return input.map((xi) => xi.mul(scale));
}
