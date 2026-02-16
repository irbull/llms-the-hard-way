/**
 * GPT Model
 *
 * Defines the model's structure: configuration, weight matrices, and
 * parameter collection. Includes the forward pass (gpt) and KV cache
 * for inference. Save/load is introduced in a later chapter.
 */

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

interface Attention {
  query: Matrix;
  key: Matrix;
  value: Matrix;
  output: Matrix;
}

interface MLP {
  hidden: Matrix;
  output: Matrix;
}

interface Layer {
  attention: Attention;
  mlp: MLP;
}

export interface Weights {
  tokenEmbedding: Matrix;
  positionEmbedding: Matrix;
  output: Matrix;
  layers: Layer[];
}

/** The trained (or untrained) model: config + weights + flattened parameter list. */
export interface Model {
  config: GPTConfig;
  weights: Weights;
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

  const weights: Weights = {
    tokenEmbedding: matrix(vocabSize, nEmbd),
    positionEmbedding: matrix(blockSize, nEmbd),
    output: matrix(vocabSize, nEmbd),
    layers: Array.from({ length: nLayer }, () => ({
      attention: {
        query: matrix(nEmbd, nEmbd),
        key: matrix(nEmbd, nEmbd),
        value: matrix(nEmbd, nEmbd),
        output: matrix(nEmbd, nEmbd),
      },
      mlp: {
        hidden: matrix(4 * nEmbd, nEmbd),
        output: matrix(nEmbd, 4 * nEmbd),
      },
    })),
  };

  const allMatrices: Matrix[] = [
    weights.tokenEmbedding,
    weights.positionEmbedding,
    weights.output,
    ...weights.layers.flatMap((layer) => [
      layer.attention.query, layer.attention.key,
      layer.attention.value, layer.attention.output,
      layer.mlp.hidden, layer.mlp.output,
    ]),
  ];
  const params = allMatrices.flatMap((mat) => mat.flatMap((row) => row));

  return { config, weights, params };
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
  const { weights } = model;

  // Step 1: Embedding lookup
  // Combine "what word is this?" with "where does it appear?" into a single
  // vector. This is the hidden state that flows through the rest of the network.
  const tokenVector: Value[] = weights.tokenEmbedding[tokenId];
  const positionVector: Value[] = weights.positionEmbedding[posId];
  let hidden: Value[] = tokenVector.map((t, i) => t.add(positionVector[i]));

  // Normalize before the first layer to keep values at a stable scale
  hidden = rmsnorm(hidden);

  // Step 2: Transformer layers
  // Each layer has two blocks: attention (gather context from other tokens)
  // followed by MLP (process the gathered information). Both use residual
  // connections so the input is added back to the output of each block.
  for (let li = 0; li < nLayer; li++) {
    const layer = weights.layers[li];

    // --- Attention block: look at previous tokens to gather context ---

    // Save the hidden state so we can add it back after the block (residual)
    const beforeAttention: Value[] = hidden;
    hidden = rmsnorm(hidden);

    // Project the hidden state into query, key, and value vectors.
    // These are structurally identical projections — training teaches them
    // to play different roles: query asks "what am I looking for?",
    // key advertises "what do I contain?", value carries "what to retrieve".
    const query: Value[] = linear(hidden, layer.attention.query);
    const key: Value[] = linear(hidden, layer.attention.key);
    const value: Value[] = linear(hidden, layer.attention.value);

    // Cache the key and value so future tokens can attend to this position
    keys[li].push(key);
    values[li].push(value);

    // Each head independently attends to a different slice of the vectors,
    // allowing the model to track multiple relationships at once
    const attentionOutput: Value[] = [];
    for (let h = 0; h < nHead; h++) {
      const headStart = h * headDim;
      const headQuery = query.slice(headStart, headStart + headDim);
      const headKeys = keys[li].map((ki) => ki.slice(headStart, headStart + headDim));
      const headValues = values[li].map((vi) => vi.slice(headStart, headStart + headDim));

      // Scaled dot-product attention: score = (query · key) / √headDim
      const attnLogits = headKeys.map((cachedKey) =>
        vsum(headQuery.map((q, j) => q.mul(cachedKey[j]))).div(Math.sqrt(headDim))
      );

      // Softmax converts scores into weights that sum to 1
      const attnWeights = softmax(attnLogits);

      // Weighted sum of value vectors — high-scoring positions contribute more
      for (let j = 0; j < headDim; j++) {
        attentionOutput.push(vsum(attnWeights.map((w, t) => w.mul(headValues[t][j]))));
      }
    }

    // Project concatenated head outputs back to the hidden dimension
    hidden = linear(attentionOutput, layer.attention.output);
    // Residual connection: add back what we had before attention
    hidden = hidden.map((h, i) => h.add(beforeAttention[i]));

    // --- MLP block: process each token's representation independently ---

    const beforeMLP: Value[] = hidden;
    hidden = rmsnorm(hidden);
    hidden = linear(hidden, layer.mlp.hidden);   // expand to 4x wider
    hidden = hidden.map((h) => h.relu());         // nonlinearity
    hidden = linear(hidden, layer.mlp.output);    // compress back
    // Residual connection: add back what we had before the MLP
    hidden = hidden.map((h, i) => h.add(beforeMLP[i]));
  }

  // Step 3: Output projection
  // Project the final hidden state to vocabulary size — one score per word.
  // These raw scores (logits) will be passed through softmax later to get
  // a probability distribution over the next token.
  return linear(hidden, weights.output);
}
