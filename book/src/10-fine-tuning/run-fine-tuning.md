# Run the Fine-Tuning

## The Fine-Tuning Script

Fine-tuning reuses the exact same `train()` function from pre-training. The only
differences are the inputs:

```typescript
// Load the ORIGINAL corpus to build the SAME tokenizer.
// The tokenizer must match the one used during pre-training,
// because the model's embeddings are tied to specific token IDs.
const originalDocs = readFileSync("data/grade1_sentences.txt", "utf-8")
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => l.trim().toLowerCase());
const tokenizer = createWordTokenizer(originalDocs);
```

This is critical. The tokenizer assigns each word an integer index by sorting all
unique words alphabetically. If we built the tokenizer from the question dataset
instead, the word "cat" might get a different index. The model's embedding for
index 89 means "cat" because that is what it learned during pre-training. Using
a different tokenizer would scramble every word.

```typescript
// Load the pre-trained model
const model = loadModel("phrases-model.json");

// Load the fine-tuning dataset
const questionDocs = readFileSync("data/questions.txt", "utf-8")
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => l.trim().toLowerCase());
```

Then we call the same `train()` function with two key changes to the
configuration:

```typescript
const fineTuned = train(
  model,
  {
    numSteps: 1000,         // 1,000 vs 5,000 for pre-training
    learningRate: 0.001,    // 0.001 vs 0.01 — 10x lower
    beta1: 0.85,
    beta2: 0.99,
    epsAdam: 1e-8,
  },
  questionDocs,
  tokenizer,
);
```

## Why Fewer Steps?

The pre-trained model already knows that "the" comes before nouns and that "cat"
is an animal. It does not need to relearn any of this. It only needs to learn
that sentences can start with "can", "do", "is", or "where" and that what
follows is a slightly different word order. 1,000 steps is enough to shift the
distribution without destroying what the model already knows.

## Why a Lower Learning Rate?

The learning rate controls how aggressively each gradient update changes the
parameters. During pre-training, we used 0.01 because the model was starting
from random noise and needed to move fast. During fine-tuning, the parameters
are already in a good place. A large learning rate would overshoot, destroying
the carefully learned representations. A 10x smaller learning rate (0.001)
nudges the model gently toward question patterns while preserving its existing
knowledge.

## Running It

```bash
npx tsx src/phrases-fine-tune.ts
```

You should see output like this:

```
vocab size: 597 (596 words + BOS)
loaded pre-trained model: 63296 params
fine-tuning sentences: 150
step    1 / 1000 | loss 2.5073
step  500 / 1000 | loss 0.9592
step 1000 / 1000 | loss 1.0530

model saved to phrases-fine-tuned-model.json
```

Notice two things:

1. **The loss starts around 2.5, not 6.39.** The pre-trained model is not
   guessing randomly. It already assigns reasonable probability to the correct
   next word, even for question patterns it has rarely seen. This head start is
   the whole point of fine-tuning.

2. **The loss drops quickly.** In pre-training, the model needed 5,000 steps to
   go from 6.39 to ~1.76. Here it reaches similar loss in just 1,000 steps because
   it already understands the building blocks.
