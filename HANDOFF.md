# Agent Handoff: LLM From Scratch Project

This document gives any new Claude agent full context on this project, Aaron's learning style, and how sessions are structured.

## What This Project Is

Aaron is working through Sebastian Raschka's "Build a Large Language Model (From Scratch)" (Manning, 2024). He's building a GPT-2 style model end-to-end in PyTorch. The companion GitHub repo for the book is at github.com/rasbt/LLMs-from-scratch.

The project lives at `~/projects/llm-from-scratch/` and is pushed to https://github.com/mccann333/llm-from-scratch (public repo).

## Current Progress (as of Chapter 4, Section 4.5)

### Completed:
- **Chapter 1** — Conceptual intro (no code)
- **Chapter 2** — Full text processing pipeline: tokenization, BPE (tiktoken), DataLoader, token embeddings, positional embeddings. File: `code/ch02_text_data.py`
- **Chapter 3** — All attention mechanisms: simplified self-attention, self-attention with trainable weights (V1/V2), causal masking, dropout, CausalAttention class, MultiHeadAttentionWrapper, efficient MultiHeadAttention. File: `code/ch03_attention.py`
- **Chapter 4 (in progress)** — GPT config dict, DummyGPTModel scaffold, LayerNorm, GELU, FeedForward, ExampleDeepNeuralNetwork with shortcut connections. File: `code/ch04_gpt_model.py`

### Next Up:
- Listing 4.6: The TransformerBlock (combines multi-head attention + feed forward + layer norm + shortcuts)
- Then the full GPTModel class that replaces the DummyGPTModel
- Then text generation

## Aaron's Learning Profile

- **Strong on architecture**, weak on syntax. He understands the big picture quickly but struggles with line-by-line code details.
- **Has dyslexia** — hand-coding is extremely painful and slow. Claude writes the code; Aaron focuses on understanding it.
- **Not a beginner** — has shipped apps to Apple App Store and Google Play Store. Knows how software projects work.
- **80/20 learner** — aims for 75-80% understanding in ~25% of the time.

## How Sessions Are Structured

### The Daily Learning Loop:
1. **Day 1 (phone Claude):** Gets 10-year-old preview of upcoming section → reads the book chapter → gets quizzed on vocabulary
2. **Day 2 (Claude Code):** Starts with a **3-question code terminology quiz** on yesterday's code → then codes the next section together

### The Two-Pass Explanation Rule:
Every concept gets explained twice:
1. **First pass — "Like I'm 10"**: Simple analogies, no jargon. Confirm he gets it.
2. **Second pass — Technical version**: Proper terminology, more detail.

### Code Terminology Quizzes:
- Give 2-3 multiple choice questions (2-3 options each) asking: "Is this a method, attribute, class, function, argument, or variable?"
- Keep it lightweight — don't slow down the book
- When he gets one wrong, give a **mnemonic or tip** to remember next time
- Key tips he's learned so far:
  - `class` keyword = class. `def` keyword = function/method.
  - `def` inside a class = method. `def` standing alone = function.
  - `self.something` = attribute (noun — thing the object HAS)
  - `def something()` = method (verb — thing the object DOES)
  - `=` inside function call parentheses = argument. `=` on its own line = variable/attribute.
  - The dot before something with `()` = method call.

### What NOT to do:
- Don't do 5 questions for every code block — one question every few blocks max
- Don't end responses with "anything else?" / "call it a night?" style wrap-up prompts
- Don't rush — understanding beats completion
- Don't explain code he didn't ask about

## Technical Setup

- **Python 3.11** (via Homebrew) in a virtual environment at `.venv/`
- **PyTorch 2.4.0** (matches the book)
- **tiktoken 0.7.0** (BPE tokenizer)
- **Mac Mini M2 Pro, 16GB** — MPS acceleration confirmed working
- **Second machine with real GPU** — available sometimes
- Device auto-detection pattern: `torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")`

## Code Files

### `code/ch02_text_data.py`
Full Chapter 2 pipeline:
- File reading and regex tokenization
- SimpleTokenizerV1 and V2 (with `<|unk|>` handling)
- BPE tokenization via tiktoken
- Sliding window data sampling
- GPTDatasetV1 class and create_dataloader_v1 function
- Token embeddings and positional embeddings

### `code/ch03_attention.py`
Full Chapter 3 attention progression:
- SelfAttention_v1 (raw nn.Parameter)
- SelfAttention_v2 (nn.Linear)
- CausalAttention (mask + dropout + batches)
- MultiHeadAttentionWrapper (multiple CausalAttention instances)
- MultiHeadAttention (efficient single-class, Listing 3.5)

### `code/ch04_gpt_model.py`
Chapter 4 GPT model building blocks (in progress):
- GPT_CONFIG_124M dictionary
- DummyGPTModel (placeholder scaffold)
- LayerNorm (Listing 4.2)
- GELU activation (Listing 4.3)
- FeedForward module (Listing 4.4)
- ExampleDeepNeuralNetwork with shortcut connections (Listing 4.5)

### `notes/ch02-glossary.md` and `notes/ch03-glossary.md`
Vocabulary and concept summaries per chapter. Aaron builds these with Claude on his phone and pastes them in.

## Book Structure Tip

The book has two types of code:
- **Numbered Listings** (e.g., "Listing 4.2") — Real building blocks that go into the final GPT model. These are keepers.
- **Unnumbered examples** — Teaching demos to illustrate a concept. Disposable. Don't go into the model.

Aaron knows this distinction. When he says "I think this is just an example," he's usually right.

## Key Concepts Aaron Has Confirmed Understanding Of:
- The full text → token → embedding → transformer pipeline
- How attention works (query/key/value, dot products, softmax, causal mask)
- Multi-head attention (multiple perspectives combined)
- Why dropout prevents overfitting
- Why layer norm stabilizes training
- Why shortcut/residual connections prevent vanishing gradients
- The difference between BPE and simple tokenizers
- How positional embeddings work (slot-based, not word-based)
- That autograd adjusts ALL learnable parameters (including embeddings)
- Batch size vs context length vs stride

## Key Concepts Still Building:
- Code terminology (method vs attribute vs function vs argument) — improving steadily
- Tensor shape reasoning — good but still developing
- The full transformer block as a single unit (next up)
