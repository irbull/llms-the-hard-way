# Generate Questions

Use the `--model` flag to generate from the fine-tuned model:

```bash
npx tsx src/phrases-generate.ts 20 --model=phrases-fine-tuned-model.json
```

Compare this with the base model:

```bash
npx tsx src/phrases-generate.ts 20
```

The fine-tuned model should produce noticeably more questions: sentences starting
with "can", "do", "is", and "where". The base model will mostly produce
declarative sentences.

The fine-tuned model will still produce some declarative sentences too. It has
not completely forgotten how to make statements; it has just shifted its
probability distribution to favor question patterns.
