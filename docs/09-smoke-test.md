# Lab 09: Smoke Test

Time to run it. In this lab you will train the model, inspect the saved
weights, and generate sentences with different sampling strategies.

## Train the Model

```bash
npx tsx src/phrases-train.ts
```

You should see output like this:

```
num sentences: 30000
vocab size: 597 (596 words + BOS)
num params: 63296
step    1 / 5000 | loss 6.3917
step  500 / 5000 | loss 3.2184
step 1000 / 5000 | loss 2.8549
step 2000 / 5000 | loss 2.4012
step 3000 / 5000 | loss 2.1538
step 4000 / 5000 | loss 1.9107
step 5000 / 5000 | loss 1.7623
model saved to phrases-model.json
```

### Verify: Loss Decreases

The loss should start near **6.39** (random guessing: `-log(1/597)`) and drop
below **2.0** by step 5000. If the loss is not decreasing, something is wrong.

### Verify: Model File Exists

```bash
ls -lh phrases-model.json
```

The file should be roughly 1.5 MB. It contains the model configuration and all
63,296 learned parameter values as JSON.

## Generate Sentences

Load the saved model and generate with default settings:

```bash
npx tsx src/phrases-generate.ts
```

```
loaded model: 63296 params, vocab 597
generating 20 sentences (temperature=0.8):

   1. the puppy wants to jump
   2. the pig swims by the park
   3. i hold the pie
   4. the fish skips
   5. the teacher is sweet
   ...
```

### Verify: Output Makes Sense

The generated sentences should:

- Start with plausible words ("the", "a", "i", "we", "mom", etc.)
- Follow basic subject-verb patterns
- Use words from the training vocabulary
- End naturally (not cut off mid-thought)

They will not be perfect. You will see occasional oddities like "the fish
quacks" or "i toes". The model has only 63K parameters and 5,000 training
steps. But the overall structure should clearly resemble English sentences.

## Try Different Sampling Strategies

### Low Temperature (Conservative)

```bash
npx tsx src/phrases-generate.ts 10 --temp=0.3
```

With low temperature, the model almost always picks the highest-probability
word. Output will be repetitive but grammatically safer.

### High Temperature (Creative)

```bash
npx tsx src/phrases-generate.ts 10 --temp=1.5
```

With high temperature, the model considers unlikely words. Output will be more
varied but may include nonsensical combinations.

### Top-k Filtering

```bash
npx tsx src/phrases-generate.ts 10 --top-k=10
```

Only the 10 most likely words can be chosen at each position. This trims the
long tail of unlikely words.

### Nucleus Sampling (Top-p)

```bash
npx tsx src/phrases-generate.ts 10 --top-p=0.9
```

Keep the smallest set of words whose probabilities sum to 0.9. Adapts to the
model's confidence at each position.

### Combined

```bash
npx tsx src/phrases-generate.ts 10 --temp=0.7 --top-p=0.9 --top-k=50
```

All three strategies working together. This is how production LLMs typically
configure their sampling.

## What You Have Built

A language model. The same architecture, at wildly different scale, behind
ChatGPT, Claude, and every other LLM.

The entire system is **~350 lines of TypeScript** across 8 files, with zero
dependencies beyond the TypeScript compiler. Every piece is explicit:

- You can read exactly how token 541 becomes a 32-dimensional vector
- You can trace exactly how attention computes which words to focus on
- You can inspect exactly how each gradient flows backward through the network
- You can see exactly how the optimizer adjusts each of the 63,296 parameters

The only difference between this and a production LLM is scale. Same concepts.
Same math. Same architecture. Just more parameters, more data, more compute,
and a few engineering optimizations to make it fast.

```
our model:         597 vocab,    63K params,  30K sentences,     minutes on a laptop
production LLM:  ~100K vocab, ~1T+ params,   trillions of tokens, months on a cluster
```

The hard part is not the size. The hard part is understanding what happens
inside, and now you do.
