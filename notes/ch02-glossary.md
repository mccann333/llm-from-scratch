# Chapter 2: Working with Text Data — Glossary

## Key Concepts

**Tokenization** — Splitting raw text into individual pieces (tokens). Can be words, subwords, or characters depending on the approach.

**Token ID** — An integer assigned to each unique token in the vocabulary. The model works with these numbers, not the actual text.

**Vocabulary** — The complete set of unique tokens the model knows about. Our simple tokenizer had 1,130 tokens; GPT-2's BPE tokenizer has 50,257.

**Encoding** — Converting text into token IDs (string to integers).

**Decoding** — Converting token IDs back into text (integers to string).

**Byte Pair Encoding (BPE)** — A tokenization algorithm that breaks words into subword pieces based on frequency. Common words stay whole ("the"), rare words get split ("sunlit" -> "sun" + "lit"). Trained on a large corpus before being used. Never produces `<|unk|>` because it can always fall back to individual characters.

**Special Tokens** — Tokens with specific roles:
- `<|endoftext|>` — Separator between unrelated documents
- `<|unk|>` — Placeholder for unknown words (used in simple tokenizers, not BPE)
- `<|pad|>` — Filler to make sequences the same length in a batch
- BOS/EOS — Beginning/end of sequence markers

**Context Length (Context Window)** — The maximum number of tokens the model can see at once. GPT-2 uses 1,024.

**Sliding Window** — The technique of moving a fixed-size window across the text to generate training samples. Each position creates an input-target pair.

**Stride** — How far the sliding window moves between samples. Stride=1 means maximum overlap; stride=max_length means no overlap.

**Batch** — Multiple training samples processed simultaneously. Batch size is how many samples per batch. Bigger batches = better GPU utilization.

**Embedding** — A vector (list of numbers) that represents a token in high-dimensional space. Each token ID maps to a unique embedding vector.

**Embedding Dimension** — How many numbers describe each token. GPT-2 Small uses 768; GPT-3 uses 12,288.

**Positional Embedding** — A second set of embeddings that encode where each token sits in the sequence. Added to the token embedding so the model knows word order.

**Input Embedding** — Token embedding + positional embedding. This is what actually gets fed into the transformer.

## The Full Pipeline

```
Raw text
  -> Tokenization (split into pieces)
  -> Token IDs (map pieces to integers)
  -> Token Embeddings (look up vectors for each ID)
  + Positional Embeddings (add position information)
  = Input Embeddings (fed into the transformer)
```

## Python Concepts Encountered

- **`with open() as f`** — File reading with automatic cleanup
- **`re.split()`** — Regex-based text splitting
- **List comprehension** — `[x for x in list if condition]`
- **`set()`** — Removes duplicates from a collection
- **`sorted()`** — Alphabetical/numerical ordering
- **Dictionary comprehension** — `{key: val for val, key in enumerate(list)}`
- **`torch.tensor()`** — Creates a PyTorch tensor
- **`nn.Embedding()`** — PyTorch's embedding lookup table
- **`Dataset` / `DataLoader`** — PyTorch classes for batching training data
