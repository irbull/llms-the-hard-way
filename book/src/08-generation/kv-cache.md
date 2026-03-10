# The KV Cache

Generation uses the same KV cache introduced in the model chapter. We
create a fresh cache before generating and pass it to every `gpt()` call:

```typescript
const { keys, values } = createKVCache(model);
```

Each call appends the current token's key and value vectors to the cache, so
the next call can attend to everything generated so far. The model never
reprocesses old tokens; it just reads their cached K/V vectors.

During training, the KV cache saves redundant work within a single sentence.
During generation, it is essential: without it, generating a 16-token sentence
would mean reprocessing the entire sequence from scratch at every position:
1 + 2 + 3 + ... + 16 = 136 forward passes instead of 16.
