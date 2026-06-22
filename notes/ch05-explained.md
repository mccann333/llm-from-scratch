# Chapter 5 — `ch05_pretraining.py` Explained (Like You're 10)

> Plain-English, block-by-block walkthrough of the Chapter 5 code.
> **Source file:** `code/ch05_pretraining.py`
> This file grows as we build Chapter 5. **So far:** Listing 5.1.

**What Chapter 5 is about:** taking the babbling, untrained brain from Chapter 4 and *teaching* it,
so the gibberish turns into real sentences. This first listing builds two little helper buttons
we'll use constantly along the way.

---

## 🧱 Block 0 — Setup: re-create the model (lines 1–23)

```python
import tiktoken
import torch
from ch04_gpt_model import GPTModel, generate_text_simple

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,   # shorter than ch04's 1024 — makes training cheaper
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()  # turn dropout off for clean text generation
```

🧒 Before we can teach the brain, we have to **build it again** (Chapter 5 starts fresh). This grabs the `GPTModel` and `generate_text_simple` we built in Chapter 4, makes the recipe card, and stamps out a brand-new (still untrained) model.

🧒 One change: `context_length` went from 1024 down to **256** — a smaller memory span so the *training* coming up is cheaper and faster to run. `model.eval()` switches off dropout so generation is clean and repeatable.

> 💡 *In-between detail:* `torch.manual_seed(123)` starts the randomness from the same spot every run, so you get the same numbers each time and nothing feels mysteriously different.

---

## 🔁 Listing 5.1 — Two helper buttons for words ⇄ numbers (lines 26–49)

```python
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add the batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove the batch dimension
    return tokenizer.decode(flat.tolist())


start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"],
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
```

🧒 Every time you want the model to write, there's a little **dance**: words → numbers → run model → numbers → words. These two **functions standing on their own** (no `class`) package that dance into two buttons so you never have to think about it again:
- **`text_to_token_ids`** = *words → the number-grid the model wants.*
- **`token_ids_to_text`** = *number-grid → words.*

🧒 **The cookie-tray idea (`unsqueeze` / `squeeze`):** the model only accepts a *stack* of sentences (a "batch"), like an oven that only takes trays, not loose cookies. So:
- `unsqueeze(0)` = **put the lone sentence on a tray** (add the batch wrapper). Shape `[4]` → `[1, 4]`.
- `squeeze(0)` = **take it back off the tray** (remove the wrapper). Shape `[1, 14]` → `[14]`.

They're mirror images — one wraps, one unwraps.

> 💡 *In-between detail:*
> - `allowed_special={"<|endoftext|>"}` tells the tokenizer to treat `<|endoftext|>` as the real end-marker, not as literal characters.
> - `.tolist()` turns the tensor grid back into a plain list of numbers, which is what `decode` wants.
> - The test at the bottom runs the full round-trip: `"Every effort moves you"` → numbers → model adds 10 tokens → back to text. Output is babble because the model is still untrained — exactly what Chapter 5 will fix.
