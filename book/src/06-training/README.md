# Training

We have data, a tokenizer, a model that can turn tokens into predictions, and
the math to compute gradients. Now we train.

Training means showing the model sentences and adjusting its parameters to make
better predictions. Each step processes one sentence through five stages:

```
"the cat eats a muffin"
    |
[Tokenize]  ->  [596, 541, 89, 152, 0, 358, 596]
    |
[Feed tokens one at a time through gpt()]
    |  position 0: feed BOS   -> predict 597 scores -> target: "the"
    |  position 1: feed "the" -> predict 597 scores -> target: "cat"
    |  position 2: feed "cat" -> predict 597 scores -> target: "eats"
    |  ...
    |
[Compute loss]  ->  how wrong was each prediction?
    |               -log(probability of the correct word)
    |
[Backward pass]  ->  loss.backward()
    |                 fills in .grad on all 63,296 parameters
    |
[Adam optimizer]  ->  nudge each parameter to reduce the loss
    |
repeat 5,000 times
```

The `gpt()` function from the model chapter handles the forward pass — turning
a token into 597 scores. This chapter covers what happens around it: measuring
how wrong those predictions are, computing gradients, and updating the weights.

Before training, the model's predictions are random noise. After 5,000 steps,
it has learned enough about English sentence structure to predict the next word
with roughly 100x better accuracy than chance.
