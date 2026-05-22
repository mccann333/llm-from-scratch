# Building a Large Language Model from Scratch

Working through Sebastian Raschka's [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) (Manning, 2024), implementing a GPT-like model end-to-end in PyTorch.

## What This Is

A hands-on learning project building a GPT-2 style language model from the ground up — tokenization, attention mechanisms, transformer architecture, pretraining, and fine-tuning. Built collaboratively with [Claude Code](https://claude.ai/claude-code).

## Progress

| Chapter | Topic | Status |
|---------|-------|--------|
| 1 | Understanding Large Language Models | Done |
| 2 | Working with Text Data | Done |
| 3 | Coding Attention Mechanisms | Done |
| 4 | Implementing a GPT Model from Scratch | In Progress |
| 5 | Pretraining on Unlabeled Data | - |
| 6 | Fine-tuning for Classification | - |
| 7 | Fine-tuning to Follow Instructions | - |

## Setup

```bash
# Python 3.11 required (PyTorch 2.4.0 compatibility)
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.4.0 tiktoken==0.7.0
```

## Hardware

- Primary: Mac Mini (M2 Pro, 16GB) with MPS acceleration
- Code auto-detects best available device (CUDA > MPS > CPU)

## Project Structure

```
code/           # Python implementations, chapter by chapter
notes/          # Glossaries, concept summaries, vocabulary lists
CLAUDE.md       # Session instructions for Claude Code
```

## Book & Resources

- Book: [Manning](https://www.manning.com/books/build-a-large-language-model-from-scratch) | [Amazon](https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167)
- Companion code: [github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- Author: Sebastian Raschka, PhD
