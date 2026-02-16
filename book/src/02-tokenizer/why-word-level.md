# Why Word-Level Tokens?

An alternative approach is character-level tokenization, where each letter is
a token. Our vocabulary would be just 28 tokens (a-z, space, plus BOS), but the
sequences become much longer: "the cat" becomes 7 character tokens instead of
2 word tokens. With only 596 unique words in our corpus, word-level tokenization
gives us short sequences that are fast to train on and easy to reason about.

Production LLMs like GPT-4 use a middle ground called [**subword tokenization**
(BPE)](https://medium.com/data-science/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0), which splits rare words into pieces while keeping common words whole.
Our word-level approach is the simplest version of this idea.
