# The Autograd Engine

Training a neural network requires answering one question for each of the
model's parameters: **if I nudge this number up a tiny bit, does the model get
better or worse?**

The answer is the parameter's **gradient**: the direction and magnitude of
change that would reduce the model's error. Computing gradients by hand for
tens of thousands of interconnected parameters would be impossible. Instead, we
use **automatic differentiation**: we build a record of every computation the
model performs, then trace backwards through that record to compute all
gradients at once.
