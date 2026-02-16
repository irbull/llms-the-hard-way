# What You Have Built

A language model. The same architecture, at wildly different scale, behind
ChatGPT, Claude, and every other LLM.

The entire system is **~450 lines of TypeScript** across 9 files, with zero
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
