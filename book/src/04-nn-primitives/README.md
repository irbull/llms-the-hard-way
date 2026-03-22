# Neural Network Primitives

On top of the autograd engine from the previous chapter, we build three operations that the
model uses throughout. These are general-purpose building blocks, the same
operations found in every neural network framework. A neural network only needs
a surprisingly small toolkit:

1. **A way to combine inputs.** Take several numbers, multiply each by a
   learned weight, and add the results together. This is just a weighted sum,
   the same idea as computing a course grade from weighted assignment scores.
   That is what `linear` does.

2. **A way to make decisions.** The model needs to express confidence: "I
   think the next word is 70% likely to be *cat* and 20% likely to be *dog*."
   Raw scores can be any number, but probabilities must be positive and sum to
   1.0. That is what `softmax` does. It turns arbitrary scores into a proper
   probability distribution.

3. **A way to stay stable.** Numbers that pass through dozens of weighted sums
   tend to either explode toward infinity or shrink toward zero. That is what
   `rmsnorm` does. It rescales a vector so the values stay in a reasonable
   range, similar to normalizing a set of exam scores to have a consistent
   spread.
