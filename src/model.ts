/**
 * GPT Model
 *
 * The transformer architecture: a function that maps input tokens to a
 * probability distribution over what comes next.
 *
 * Follows GPT-2 with minor simplifications:
 * - RMSNorm instead of LayerNorm
 * - No biases
 * - ReLU instead of GeLU
 *
 * Config: 16 embedding dims, 4 attention heads, 1 layer, 16 max context
 * → 4,192 parameters total
 */

import { readFileSync, writeFileSync } from "node:fs";
import { Value, vsum } from "./autograd.js";
import { type Matrix, linear, softmax, rmsnorm } from "./nn.js";
import { gauss } from "./rng.js";

// --- Configuration ---

export interface GPTConfig {
  nLayer: number;
  nEmbd: number;
  blockSize: number;
  nHead: number;
  headDim: number;
  vocabSize: number;
}

export type StateDict = Record<string, Matrix>;

/** The trained (or untrained) model: config + weights + flattened parameter list. */
export interface Model {
  config: GPTConfig;
  stateDict: StateDict;
  params: Value[];
}

// --- Model Creation ---

function matrix(nout: number, nin: number, std = 0.08): Matrix {
  return Array.from({ length: nout }, () =>
    Array.from({ length: nin }, () => new Value(gauss(0, std)))
  );
}

/** Create a new model with randomly initialized weights. */
export function createModel(config: GPTConfig): Model {
  const { nEmbd, nLayer, blockSize, vocabSize } = config;

  const stateDict: StateDict = {
    wte: matrix(vocabSize, nEmbd),
    wpe: matrix(blockSize, nEmbd),
    lm_head: matrix(vocabSize, nEmbd),
  };

  for (let i = 0; i < nLayer; i++) {
    stateDict[`layer${i}.attn_wq`] = matrix(nEmbd, nEmbd);
    stateDict[`layer${i}.attn_wk`] = matrix(nEmbd, nEmbd);
    stateDict[`layer${i}.attn_wv`] = matrix(nEmbd, nEmbd);
    stateDict[`layer${i}.attn_wo`] = matrix(nEmbd, nEmbd);
    stateDict[`layer${i}.mlp_fc1`] = matrix(4 * nEmbd, nEmbd);
    stateDict[`layer${i}.mlp_fc2`] = matrix(nEmbd, 4 * nEmbd);
  }

  const params: Value[] = Object.values(stateDict).flatMap((mat) =>
    mat.flatMap((row) => row)
  );

  return { config, stateDict, params };
}

// --- Save / Load ---

/** Save a trained model to a JSON file (config + parameter values). */
export function saveModel(model: Model, path: string): void {
  const data = {
    config: model.config,
    weights: model.params.map((p) => p.data),
  };
  writeFileSync(path, JSON.stringify(data));
}

/** Load a model from a JSON file. Recreates the structure, then fills in the learned weights. */
export function loadModel(path: string): Model {
  const data = JSON.parse(readFileSync(path, "utf-8"));
  const model = createModel(data.config);
  for (let i = 0; i < model.params.length; i++) {
    model.params[i].data = data.weights[i];
  }
  return model;
}

// --- KV Cache ---

/** Create fresh key/value caches for a new sequence. Must be called per-sequence. */
export function createKVCache(model: Model): {
  keys: Value[][][];
  values: Value[][][];
} {
  return {
    keys: Array.from({ length: model.config.nLayer }, () => []),
    values: Array.from({ length: model.config.nLayer }, () => []),
  };
}

// --- Forward Pass ---

/**
 * Run one step of the GPT: given a token at a position, return logits
 * over the vocabulary for the next token.
 *
 * The keys/values caches are mutated (appended to) on each call —
 * this is the KV cache that avoids recomputing attention for past positions.
 */
export function gpt(
  model: Model,
  tokenId: number,
  posId: number,
  keys: Value[][][],
  values: Value[][][],
): Value[] {
  const { nLayer, nHead, headDim } = model.config;
  const { stateDict } = model;

  const tokEmb = stateDict["wte"][tokenId]; // token embedding
  const posEmb = stateDict["wpe"][posId]; // position embedding
  let x = tokEmb.map((t, i) => t.add(posEmb[i])); // combined embedding

  x = rmsnorm(x);

  for (let li = 0; li < nLayer; li++) {
    // 1) Multi-head Attention block
    const xResidual1 = x;
    x = rmsnorm(x);
    const q = linear(x, stateDict[`layer${li}.attn_wq`]);
    const k = linear(x, stateDict[`layer${li}.attn_wk`]);
    const v = linear(x, stateDict[`layer${li}.attn_wv`]);
    keys[li].push(k);
    values[li].push(v);

    const xAttn: Value[] = [];
    for (let h = 0; h < nHead; h++) {
      const hs = h * headDim;
      const qH = q.slice(hs, hs + headDim);
      const kH = keys[li].map((ki) => ki.slice(hs, hs + headDim));
      const vH = values[li].map((vi) => vi.slice(hs, hs + headDim));
      const attnLogits = kH.map((kT) =>
        vsum(qH.map((qj, j) => qj.mul(kT[j]))).div(Math.sqrt(headDim))
      );
      const attnWeights = softmax(attnLogits);
      for (let j = 0; j < headDim; j++) {
        xAttn.push(vsum(attnWeights.map((w, t) => w.mul(vH[t][j]))));
      }
    }

    x = linear(xAttn, stateDict[`layer${li}.attn_wo`]);
    x = x.map((a, i) => a.add(xResidual1[i]));

    // 2) MLP block
    const xResidual2 = x;
    x = rmsnorm(x);
    x = linear(x, stateDict[`layer${li}.mlp_fc1`]);
    x = x.map((xi) => xi.relu());
    x = linear(x, stateDict[`layer${li}.mlp_fc2`]);
    x = x.map((a, i) => a.add(xResidual2[i]));
  }

  return linear(x, stateDict["lm_head"]);
}
