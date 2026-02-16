# The Question Dataset

Our vocabulary has 596 words. We cannot introduce new words without resizing the
embedding tables, so the question dataset must use only words the model already
knows. This is realistic: real fine-tuning rarely changes the tokenizer.

The vocabulary includes four question words: `can`, `do`, `is`, and `where`.
That is enough for four question patterns:

```
can the cat run
do the birds fly
is the dog big
where is the ball
```

The file `data/questions.txt` contains 150 sentences following these patterns.
Each sentence uses only words from the existing vocabulary.
