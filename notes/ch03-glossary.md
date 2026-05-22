# Chapter 3: Coding Attention Mechanisms — Glossary

## Key Concepts

**Self-Attention** — A mechanism where every token in a sequence looks at every other token in the same sequence to determine relevance. "Self" because it's attending to its own input, not something external.

**Dot Product** — Multiplying two vectors element-by-element and summing the results. Used to measure similarity between tokens. Large dot product = highly relevant; small = not relevant.

**Attention Scores** — Raw dot product values between Query and Key vectors. These are the unprocessed similarity measurements before normalization.

**Attention Weights** — Attention scores after softmax normalization. Each row sums to 1, representing a probability distribution of how much each token attends to every other token.

**Softmax** — A function that converts raw scores into probabilities that sum to 1. Larger inputs get larger probabilities; negative infinity becomes 0.

**Scaling (sqrt(d_k))** — Dividing attention scores by the square root of the key dimension before softmax. Prevents scores from getting too large, which would make softmax output extreme (all attention on one token). Keeps attention spread across multiple relevant tokens.

**Query (Q)** — "What am I looking for?" Each token's query vector represents what information it needs.

**Key (K)** — "What do I have to offer?" Each token's key vector advertises what it contains.

**Value (V)** — "What do I actually carry?" The content that gets passed along, weighted by attention scores.

**Weight Matrices (W_query, W_key, W_value)** — Learned linear transformations that convert input embeddings into Query, Key, and Value vectors. These are the trainable parameters of attention.

**Context Vector** — The output of attention for each token. A weighted sum of all Value vectors, where the weights are the attention weights. Represents the token enriched with information from relevant tokens.

**Causal Attention (Masked Self-Attention)** — Self-attention with a restriction: each token can only attend to tokens at the same or earlier positions. Prevents "peeking at the future." Essential for autoregressive (generative) models.

**Causal Mask** — A triangular matrix used to block future positions. Applied as negative infinity before softmax, which converts those positions to zero probability.

**Dropout** — Randomly zeroing out some attention connections during training. Prevents overfitting by forcing the model not to rely on any single connection. Surviving values are scaled up to compensate.

**Multi-Head Attention** — Running multiple attention mechanisms in parallel, each with its own Q/K/V weight matrices. Each "head" can focus on different types of relationships (grammar, meaning, proximity, etc.). Results are combined at the end.

**Head Dimension** — The embedding dimension divided by number of heads. With 768-dim embeddings and 12 heads, each head works with 64 dimensions.

**nn.Module** — PyTorch's base class for all neural network components. Gives classes the ability to track parameters, move to GPU, save/load weights, etc.

**nn.Linear** — A learned linear transformation (matrix multiplication + optional bias). The "proper" PyTorch way to implement weight matrices.

**nn.Parameter** — A tensor that PyTorch knows to treat as a trainable weight. Gets updated during backpropagation.

**register_buffer** — Stores a tensor with the model (moves to GPU, gets saved/loaded) but does NOT get trained. Used for the causal mask.

**Transpose (.T or .transpose())** — Flipping a matrix's rows and columns. Needed to align dimensions for matrix multiplication, especially Q @ K^T.

## Architecture Progression

```
Simplified Self-Attention (no learning, raw dot products)
  -> Self-Attention V1 (trainable weights via nn.Parameter)
  -> Self-Attention V2 (trainable weights via nn.Linear)
  -> Causal Attention (added mask + dropout)
  -> Multi-Head Attention Wrapper (multiple CausalAttention instances)
  -> Multi-Head Attention (efficient single-class implementation)
```

## Python Concepts Encountered

- **`class X(nn.Module)`** — Defining a PyTorch neural network component
- **`super().__init__()`** — Initializing the parent class
- **`def __init__`** — Constructor method, runs when creating an instance
- **`def forward`** — The method PyTorch calls when you pass data through the model
- **`@` operator** — Matrix multiplication in Python/PyTorch
- **`torch.triu` / `torch.tril`** — Upper/lower triangular matrices
- **`masked_fill_`** — In-place replacement of values (trailing underscore = in-place)
- **`torch.softmax()`** — Normalizes values into probabilities summing to 1
- **`nn.ModuleList`** — A list of nn.Module objects that PyTorch can track
- **`torch.cat()`** — Concatenates tensors along a dimension
- **`assert`** — A safety check that raises an error if the condition is false
