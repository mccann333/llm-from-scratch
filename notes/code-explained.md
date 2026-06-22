# Code Explained — "Like You're 10" Walkthroughs

Plain-English, block-by-block walkthroughs of every code file we build for Sebastian Raschka's
*Build a Large Language Model (From Scratch)*. The goal: by the time the book is done, every line
of code in this project has a friendly explanation you can re-read any time.

## Index

| Chapter | Code file | Walkthrough |
|---|---|---|
| 2 — Working with Text Data | `code/ch02_text_data.py` | [ch02-explained.md](ch02-explained.md) ✅ |
| 3 — Coding Attention Mechanisms | `code/ch03_attention.py` | _coming next_ ⏳ |
| 4 — Implementing a GPT Model | `code/ch04_gpt_model.py` | [ch04-explained.md](ch04-explained.md) ✅ |
| 5 — Pretraining on Unlabeled Data | `code/ch05_pretraining.py` | [ch05-explained.md](ch05-explained.md) 🚧 _in progress (Listing 5.1)_ |

## How these are written (the format)

So every doc stays consistent as the project grows:

- **One doc per code file**, named `chXX-explained.md`.
- **Block by block.** Show the code block, then explain it. Walk the file top to bottom.
- **Explain like you're 10.** Use everyday analogies. Mark kid-level explanations with 🧒.
- **In-between details.** Extra notes (what a keyword means, a gotcha) go in a `> 💡` quote.
- **No grammar jargon.** Don't lean on "noun/verb/adjective." Use **HAS vs DOES**:
  - *Attribute* = a thing the object **HAS** (a stored value) → has a `self.` in front, no `()` after.
  - *Method / function* = a thing being **DONE** (an action) → has `()` after it.
  - *Argument* = `name=value` inside a call's `( )`.
  - *Variable* = a bare `name =` with no owner-dot in front.
- **Call out keepers vs. throwaways** — note when a code block is just a teaching demo, not part of the final model.

## Workflow

Each time we finish building a code file, add (or update) its `chXX-explained.md` here and commit it
to the repo so it syncs across machines.
