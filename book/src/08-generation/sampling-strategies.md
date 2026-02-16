# Comparing Sampling Strategies

| Strategy | What it does | Trade-off |
|---|---|---|
| Temperature | Reshapes the entire distribution | Simple, but low-probability junk words can still be picked |
| Top-k | Hard cutoff at *k* words | Fixed *k* does not adapt to model confidence |
| Top-p | Adaptive cutoff by cumulative probability | Handles both confident and uncertain positions well |

In practice, these are combined. A typical production configuration might use
`temperature=0.7`, `top_p=0.9`, and `top_k=50` together. Temperature reshapes
the distribution, then top-k and top-p trim the tail. Our implementation
supports all three, individually or combined.
