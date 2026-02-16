# Train the Model

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

## Verify: Loss Decreases

The loss should start near **6.39** (random guessing: `-log(1/597)`) and drop
below **2.0** by step 5000. If the loss is not decreasing, something is wrong.

## Verify: Model File Exists

```bash
ls -lh phrases-model.json
```

The file should be roughly 1.5 MB. It contains the model configuration and all
63,296 learned parameter values as JSON.
