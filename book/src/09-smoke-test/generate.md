# Generate Sentences

Load the saved model and generate with default settings:

```bash
npx tsx src/phrases-generate.ts
```

```
loaded phrases-model.json: 63296 params, vocab 597
generating 20 sentences (temperature=0.8):

   1. the puppy wants to jump
   2. the pig swims by the park
   3. i hold the pie
   4. the fish skips
   5. the teacher is sweet
   ...
```

## Verify: Output Makes Sense

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
