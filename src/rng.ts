/**
 * Seeded Random Number Generator
 *
 * JavaScript has no built-in seeded RNG, so we roll our own:
 * - mulberry32 for uniform random numbers
 * - Box-Muller transform for Gaussian (normal) distribution
 * - Fisher-Yates for array shuffling
 * - Weighted random choice for sampling from probability distributions
 */

let _rngState = 0;

export function seed(s: number): void {
  _rngState = s | 0;
}

/** Mulberry32: a fast, seedable 32-bit PRNG. Returns a uniform float in [0, 1). */
export function random(): number {
  _rngState = (_rngState + 0x6d2b79f5) | 0;
  let t = _rngState;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

/** Box-Muller transform: convert two uniform samples into a Gaussian sample. */
export function gauss(mean: number, std: number): number {
  let u1 = random();
  while (u1 === 0) u1 = random();
  const u2 = random();
  return mean + std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/** Fisher-Yates shuffle: randomly reorder an array in-place. */
export function shuffle<T>(arr: T[]): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

/** Sample an index from a weighted distribution. Used for token sampling during inference. */
export function weightedChoice(weights: number[]): number {
  const total = weights.reduce((a, b) => a + b, 0);
  let r = random() * total;
  for (let i = 0; i < weights.length; i++) {
    r -= weights[i];
    if (r <= 0) return i;
  }
  return weights.length - 1;
}
