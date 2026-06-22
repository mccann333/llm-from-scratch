# Chapter 4 — `ch04_gpt_model.py` Explained (Like You're 10)

> Plain-English, block-by-block walkthrough of the Chapter 4 code — the complete GPT "brain."
> **Source file:** `code/ch04_gpt_model.py`

**Where this fits in the pipeline:**
- **ch02 (the word-to-number machine):** chops text into word-pieces, gives each an ID, and a "meaning card" of numbers.
- **ch03 (the attention machine):** lets every word look around at the other words and decide which matter.
- **ch04 (this file — the whole brain):** bolts it all together into a working GPT.

---

## 🧰 Block 0 — The top of the file (lines 1–16)

```python
"""Chapter 4: Implementing a GPT Model from Scratch"""

import torch
import torch.nn as nn

from ch03_attention import MultiHeadAttention

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

🧒 **The `import` lines** are like grabbing your toolboxes off the shelf. `torch` is the big math toolbox; `nn` is the "neural network parts" toolbox; line 6 reaches into your *own* Chapter 3 file and grabs the attention machine you already built.

🧒 **`GPT_CONFIG_124M` is the recipe card** — one box holding all the settings, so every part can peek at it instead of memorizing numbers:

| Setting | Kid translation |
|---|---|
| `vocab_size: 50257` | How many different word-pieces the model knows — its whole dictionary. |
| `context_length: 1024` | How many word-pieces it can hold in its head at once (its memory span). |
| `emb_dim: 768` | How many numbers describe *one* word-piece's meaning (768 tiny measurements). |
| `n_heads: 12` | How many "points of view" the attention looks from at the same time. |
| `n_layers: 12` | How many workstations are stacked up — how *deep* the brain is. |
| `drop_rate: 0.1` | How often it randomly throws away 10% of its notes so it can't cheat by memorizing. |
| `qkv_bias: False` | A tiny on/off switch in the attention math. Off, to match the real GPT-2. |

> 💡 *In-between detail:* "124M" means **124 million** little adjustable numbers inside. Those numbers are what *learning* changes. Before training they're random — that's why the model babbles. Chapter 5 fixes that.

---

## 🧪 Block 1 — The rough draft (lines 19–58)

```python
class DummyGPTModel(nn.Module): ...
class DummyTransformerBlock(nn.Module): ...
class DummyLayerNorm(nn.Module): ...
```

🧒 These three "**Dummy**" parts are a **cardboard mockup** — a rough draft where the important pieces are *fake*. The `DummyTransformerBlock` just hands the data straight back (`return x`), like a vending machine that gives your coin back.

🧒 **Why build fakes first?** So you can check the *shape* of the whole machine — does data flow in one end and out the other? — before building the hard, real parts. Everything below this block replaces a cardboard part with a real one.

> 💡 *In-between detail:* **`class … (nn.Module)`** means "start with all of PyTorch's brain-machinery, then add my stuff." Inside each part, **`__init__`** = *unpack the box and lay the parts on the table* (what the part **HAS**), and **`forward`** = *turn the machine on and run stuff through it* (what the part **DOES**). Same HAS-vs-DOES split in every block.

---

## ⚖️ Block 2 — `LayerNorm`, the leveler (lines 61–73)

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

🧒 **The problem it solves:** as numbers flow through a deep brain, some get HUGE and some get tiny, making learning wobbly — like music where some parts are ear-splitting and some too quiet to hear.

🧒 **What it DOES:** takes a row of numbers and **re-levels them** into a calm range (centered near zero, not too spread out). Then `scale` and `shift` are two **volume knobs** the model can learn to turn, in case it *wants* some things louder again.

> 💡 *In-between detail:* `self.eps` is a teeny number (0.00001) that prevents a "divide by zero" crash. `self.scale`/`self.shift` are `nn.Parameter`s — **numbers the model is allowed to learn**. `scale` starts at all 1s (don't change volume), `shift` at all 0s (don't nudge) — neutral starting points.

---

## 🎚️ Block 3 — `GELU`, the smooth switch (lines 76–85)

```python
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi))
            * (x + 0.044715 * torch.pow(x, 3))
        ))
```

🧒 A brain needs a **decision step** — "let this signal through, or not?" Without it, the whole model is one boring straight line that can't learn interesting things.

🧒 `GELU` is a **soft on/off dimmer switch.** Big positives pass through almost fully, very negatives get mostly blocked, and in-between ones get *partly* let through. That smoothness lets the model learn curvy patterns, not just straight lines.

> 💡 *In-between detail:* The scary formula is just the recipe for the smooth curve. You only need its **job**: "let signals through, smoothly." (This class has **no `__init__` parts to store** — it only *DOES*, it doesn't *HAVE*.)

---

## 🛠️ Block 4 — `FeedForward`, the think-it-over station (lines 88–99)

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
```

🧒 After the words look at each other (attention), each word goes off **by itself to think**. That's this part. The trick is the **"expand, think, shrink"** sandwich:
1. `nn.Linear(768 → 3072)` — **blow the idea up big** (4× wider) for room to work.
2. `GELU()` — **smooth decision step** in that roomy space.
3. `nn.Linear(3072 → 768)` — **squeeze it back down** to fit the next part.

Like unfolding a big map to study it, then folding it back into your pocket.

> 💡 *In-between detail:* `nn.Sequential` = "do these steps in order." `nn.Linear` is a "mixing layer" that blends its input numbers into new combinations (lots of learnable numbers live here).

---

## 🔗 Block 5 — `ExampleDeepNeuralNetwork`, the *throwaway* demo (lines 102–122)

```python
class ExampleDeepNeuralNetwork(nn.Module):
    ...
    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x
```

🧒 **This one is NOT part of the real model** — it's a demo whose only job is to *show why a trick works.* (This is the chapter's one throwaway; everything else is a keeper.)

🧒 **The trick — "shortcut connections":** in a very deep brain, the learning signal can fade to nothing on the way back, like a whisper down a long line of people. The fix is a **shortcut wire** (`x = x + layer_output`) that lets the original message skip ahead so it never gets lost.

> 💡 *In-between detail:* You'll see this exact `x = x + something` move show up for real in the very next block — that's the whole point of the demo.

---

## 🏭 Block 6 — `TransformerBlock`, one full workstation (lines 125–157)

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(...)      # Station 1: "everybody talk"
        self.ff = FeedForward(cfg)              # Station 2: "think privately"
        self.norm1 = LayerNorm(cfg["emb_dim"])  # leveler before Station 1
        self.norm2 = LayerNorm(cfg["emb_dim"])  # leveler before Station 2
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut          # add the original back

        # Feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut          # add the original back
        return x
```

🧒 This is **one complete workstation** snapping together everything above. It has **two mini-stations**, each running the *exact same five-step rhythm:*

> **save a copy → level the numbers → do the work → throw away 10% (dropout) → add the saved copy back.**

🧒 Station 1's work is **attention** (words talk to each other). Station 2's work is **feed-forward** (each word thinks alone). `x = x + shortcut` is the **shortcut wire** the demo taught you — it protects information so nothing important is lost.

> 💡 *In-between detail:* `shortcut = x` = *"keep a copy of what came in"* to re-add at the end. `drop_shortcut` is **dropout** — randomly blanking 10% of numbers so the model can't lazily memorize. Same numbers in, same shape out — which is exactly why we can stack a bunch of these in a row.

---

## 🧠 Block 7 — `GPTModel`, the whole factory (lines 160–184)

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

🧒 This is the **whole brain** — just the cardboard mockup from Block 1 with the *real* parts dropped in. The parts it HAS:
- `tok_emb` — the **dictionary** turning each word-number into its 768-number meaning card.
- `pos_emb` — position cards saying *"you're the 1st word, you're the 2nd word…"* so the model knows **order** ("dog bites man" ≠ "man bites dog").
- `trf_blocks` — **12 workstations stacked in a row** (the `for _ in range(n_layers)` stamps out 12 copies).
- `final_norm` — one last leveling.
- `out_head` — the **cashier at the end** who turns the model's thinking into a score for every possible next word.

🧒 What it DOES (the `forward` assembly line):
1. Look up each word's meaning card **and** its position card, and **add them** so each word carries *both* "what I mean" and "where I am."
2. Sprinkle dropout.
3. Push through all **12 workstations**.
4. One final leveling.
5. The cashier (`out_head`) gives a score for all 50,257 possible next words.

> 💡 *In-between detail:* `device=in_idx.device` makes sure the position cards are built **in the same place** (same CPU or GPU) as the word cards, so they can be added. `logits` are **raw scores, not yet decisions** — turning scores into a chosen word is the next block's job.

---

## ✍️ Block 8 — `generate_text_simple`, the writer (lines 187–213)

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

🧒 This is the **word-prediction game** that makes the model write. Notice it's a **function standing on its own** (no `class` around it) — a helper that *uses* the model, rather than a part *inside* it.

🧒 Round by round:
1. **`idx_cond = …`** — if the sentence got longer than the model's memory span, keep only the **most recent** chunk.
2. **`with torch.no_grad(): logits = model(idx_cond)`** — ask the model for its scores. `no_grad` = *"don't take learning-notes right now, we're just using you"* (faster).
3. **`logits = logits[:, -1, :]`** — we only care about the **next word**, so grab just the last position's scores.
4. **`softmax`** — turn raw scores into **percentages**.
5. **`argmax`** — **point at the single biggest** percentage and pick that word.
6. **`torch.cat`** — **glue** the new word onto the end of the sentence.
7. Loop: feed the longer sentence back in and do it again.

> 💡 *In-between detail:* This is **autoregressive generation** — fancy words for "it eats its own output." Each new word becomes part of the input for the next guess. That's the whole secret behind how GPT writes: one word at a time, always feeding itself back.

---

**The entire brain:** word-cards in → 12 workstations of looking-around and thinking → a guessed next word out, over and over. Untrained = babble; Chapter 5 training is what turns the babble into real sentences (same code, just changing those 124 million numbers).
