# Residual Connections

After each block (attention and MLP), the original input is added back:

```typescript
hidden = hidden.map((h, i) => h.add(beforeBlock[i]));
```

where `beforeBlock` is the hidden state saved before the block started. This
is element-wise addition. The output of each block is *added to* the input
that went in, not substituted for it. The consequence: a block doesn't
have to reproduce the entire representation from scratch. It only needs to
learn a useful *correction* — a delta to add on top of what's already there.
If a block has nothing useful to contribute, its weights can settle near zero
and the input passes through unchanged.

## Why This Matters for Training

During training, the model adjusts weights using gradients — signals that flow
backward through the network saying "this weight should go up" or "this weight
should go down." Without residual connections, those signals have to pass
through every layer's transformations on the way back. Each layer can shrink
the signal (vanishing gradients) or amplify it out of control (exploding
gradients). By the time the signal reaches the early layers, it's often
degraded beyond usefulness.

The residual connection gives gradients a direct path. Because the input is
added to the output (`y = f(x) + x`), the gradient of `y` with respect to `x`
always includes a term of 1 — the identity. No matter what `f` does to the
gradient, the skip connection ensures the raw signal can still flow backward
unimpeded. This is what makes training deep networks practical.
