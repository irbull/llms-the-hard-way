# Catastrophic Forgetting

What happens if we fine-tune too aggressively? Try increasing the learning rate
or step count:

```typescript
{ numSteps: 5000, learningRate: 0.01, ... }  // same as pre-training
```

With these settings, the model will almost exclusively produce questions and
lose its ability to generate varied declarative sentences. This is called
**catastrophic forgetting**: the new data completely overwrites the old
knowledge.

This is one of the fundamental tensions in fine-tuning. You want the model to
learn the new task, but you do not want it to forget everything else. In
practice, this is managed by:

- **Low learning rates** (what we did here)
- **Few training steps** (what we did here)
- **Freezing layers** (not updating early transformer layers, only the later
  ones)
- **LoRA** (adding small adapter matrices instead of modifying the original
  weights)

Our approach is the simplest version: just be gentle with the learning rate and
stop early. It works well for small models and small distribution shifts.

## What This Teaches

Fine-tuning is the most common way LLMs are deployed in the real world:

```
Pre-training (expensive, done once):
  trillions of tokens  ->  base model

Fine-tuning (cheap, done many times):
  base model + small dataset  ->  specialized model
```

GPT-4 is fine-tuned for conversation. Code Llama is fine-tuned for programming.
Med-PaLM is fine-tuned for medical questions. Each starts from the same kind of
base model and adapts it with a small, targeted dataset.

The entire fine-tuning pipeline here is five lines of setup and one call to
`train()`. The same function, the same optimizer, the same math. The only
differences are the data and the hyperparameters. That simplicity is not a
shortcut. It is how fine-tuning actually works.
