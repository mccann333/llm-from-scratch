# Chapter 3 — `ch03_attention.py` Explained (Like You're 10)

> Plain-English, block-by-block walkthrough of the Chapter 3 code — the attention machine.
> **Source file:** `code/ch03_attention.py`

**Read this first — Chapter 3 is a LADDER.** It builds attention **five times**, and each version adds
*one* new idea on top of the last. Only the **top rung** (`MultiHeadAttention`, Listing 3.5) actually
goes into the real GPT model — that's the one `ch04` imports. The earlier four are how you *learn to
climb*: keep them for understanding, but they get superseded.

| Rung | Class | New idea it adds | Keeper? |
|---|---|---|---|
| 1 | `SelfAttention_v1` | attention, built by hand | stepping-stone |
| 2 | `SelfAttention_v2` | same, but with tidy `nn.Linear` | stepping-stone |
| 3 | `CausalAttention` | can't peek at future words + dropout + batches | stepping-stone |
| 4 | `MultiHeadAttentionWrapper` | several attention machines at once (easy way) | stepping-stone |
| 5 | `MultiHeadAttention` | several at once (fast way) | ✅ **the keeper** |

---

## 🧒 Big picture: what is "attention"?

Words mean different things depending on their neighbors ("**bank**" by a *river* vs. by *money*).
Attention lets every word **look at all the other words** and pull in the ones that help explain it.

Picture a **classroom**. For each word:
- **Query** = the question it asks the room: *"who here is relevant to me?"*
- **Key** = the label each word holds up: *"here's what I'm about."*
- **Value** = the actual content each word can share.

A word compares its **Query** against everyone's **Keys** to decide *who to listen to*, then grabs a
**blend of everyone's Values**, weighted by how relevant each one is. That blend is the word's new,
context-aware meaning (a "context vector").

Two helper moves you'll see everywhere:
- **`@`** means **matrix multiply** — the big grid-math that compares/blends whole rows of numbers at once.
- **softmax** turns raw match-scores into **percentages that add to 100%** ("listen 70% to this word, 20% to that…").

---

## 🧰 Block 0 — Imports (lines 1–4)

```python
import torch
import torch.nn as nn
```

🧒 Grab the math toolbox (`torch`) and the neural-network parts toolbox (`nn`). Same two you use in every file.

---

## 🪜 Rung 1 — `SelfAttention_v1` (Listing 3.1, lines 7–22)

```python
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.w_query = nn.Parameter(torch.rand(d_in, d_out))
        self.w_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.w_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.w_key
        queries = x @ self.w_query
        values = x @ self.w_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec
```

🧒 The **built-by-hand** version. The things it HAS are **three grids of numbers** (`w_query`, `w_key`, `w_value`) — these turn each word into its Query, its Key, and its Value.

🧒 What it DOES, step by step:
1. Multiply the words by each grid to make `keys`, `queries`, `values`.
2. `queries @ keys.T` — **every word's question compared to every word's label** → a table of match-scores.
3. `softmax(... / …**0.5)` — turn those scores into **listening percentages**.
4. `attn_weights @ values` — **blend everyone's Values** by those percentages → the new meanings.

> 💡 *In-between detail:* `nn.Parameter` = "a grid of numbers the model is allowed to learn." `keys.T` is the grid **flipped** (rows↔columns) so the multiply lines up. The `/ …**0.5` is **scaling** — dividing by the square root of the size — which keeps the scores from getting so big that softmax gets lopsided.

---

### The sample input (lines 25–39)

```python
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your
     [0.55, 0.87, 0.66],  # journey
     [0.57, 0.85, 0.64],  # starts
     [0.22, 0.58, 0.33],  # with
     [0.77, 0.25, 0.10],  # one
     [0.05, 0.80, 0.55]]  # step
)
sa_v1 = SelfAttention_v1(d_in=3, d_out=2)
print(sa_v1(inputs))
```

🧒 A tiny pretend sentence — **6 words, each described by 3 numbers** — used to test every version in this file. It's the "Your journey starts with one step" example. This is just practice data so you can watch the machine work.

---

## 🪜 Rung 2 — `SelfAttention_v2` (Listing 3.2, lines 42–57)

```python
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec
```

🧒 **Exact same idea as Rung 1** — only the three hand-made grids are swapped for **`nn.Linear`** "mixing layers." `nn.Linear` is just a tidier, smarter-starting way to do the same transformation.

🧒 Spot the difference in `forward`: `self.W_key(x)` (a **call**, with `()`) instead of `x @ self.w_key` (a manual multiply). The `nn.Linear` does the multiply for you when you call it.

> 💡 *In-between detail:* better starting numbers = trains more smoothly. That's the whole reason for the swap; the attention math is identical.

---

## 🪜 Rung 3 — `CausalAttention` (Listing 3.3, lines 65–91)

```python
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec
```

🧒 Now it becomes a **real** attention machine, with **three upgrades**:

1. **The mask — no peeking at the future.** When the model writes, the next words don't exist yet, so a word is only allowed to look at **itself and the words before it**. The mask blocks the "future" spots.
2. **Dropout** on the listening-percentages, so the model can't over-rely on any one connection.
3. **Batches** — it now handles a whole **stack of sentences** at once (`b` = how many).

> 💡 *In-between detail:*
> - `torch.triu(..., diagonal=1)` builds a triangle of 1s marking the "future" spots to block; `masked_fill_(..., -torch.inf)` sets those scores to **negative infinity**, so after softmax they become **0%** (fully ignored).
> - `register_buffer("mask", ...)` stores the mask as part of the model but as a **fixed helper that never learns** (and it travels to CPU/GPU with the model).
> - `keys.transpose(1, 2)` is the batch-friendly version of "flip the grid."

---

## 🪜 Rung 4 — `MultiHeadAttentionWrapper` (lines 104–114)

```python
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
```

🧒 **"Multi-head"** = run attention **several times in parallel**, each "head" free to notice different things (one head might track *who did what*, another might track *the topic*). More points of view = richer understanding.

🧒 This is the **easy but slower** way: literally make `num_heads` separate `CausalAttention` machines, run them all, and **glue their answers side by side** (`torch.cat(..., dim=-1)`).

> 💡 *In-between detail:* `nn.ModuleList` is just "a list of parts PyTorch keeps track of." The list-comprehension `[CausalAttention(...) for _ in range(num_heads)]` stamps out one machine per head — same trick you later use to stack 12 transformer blocks in ch04.

---

## 🪜 Rung 5 — `MultiHeadAttention` (Listing 3.5, lines 128–178) ✅ THE KEEPER

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        # split into heads
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # recombine heads
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec
```

🧒 **The same multi-head idea as Rung 4, but done in ONE smart machine instead of many separate ones — much faster.** This is the exact class your `TransformerBlock` uses in ch04.

🧒 The clever trick: instead of building a separate machine per head, it makes the Query/Key/Value **once** (big), then **slices** them into `num_heads` pieces, does the attention for **all heads at the same time** with one batch of grid-math, **stitches** the pieces back together, and runs a final **mixing layer** (`out_proj`).

🧒 Walk-through of `forward`:
1. Make big `keys`, `queries`, `values`.
2. `.view(...)` — **cut** each into `num_heads` slices (that's the "split into heads").
3. `.transpose(...)` — **line the grids up** so all heads can be computed together.
4. `queries @ keys.transpose(2,3)` → match-scores; `masked_fill_` blocks the future; `softmax` → percentages; `dropout`.
5. Blend the Values, `.transpose` + `.contiguous().view(...)` to **glue the heads back into one**, then `out_proj` mixes them into the final answer.

> 💡 *In-between detail:*
> - `assert d_out % num_heads == 0` — a **safety check**: the width must divide evenly into the heads, or it stops with a clear message.
> - `head_dim = d_out // num_heads` — how wide each head's slice is.
> - `.view(...)` reshapes without changing the numbers (just re-groups them); `.contiguous()` tidies the memory first so `.view` is allowed.
> - `out_proj` is a final `nn.Linear` that blends the heads' outputs together — the wrapper (Rung 4) didn't have this; it's part of what makes this version the real deal.

---

**The whole ladder in one line:** attention by hand → attention with `nn.Linear` → add the no-peeking mask + dropout + batches → run many heads (easy way) → run many heads (fast way). Only that last rung, `MultiHeadAttention`, climbs into the real model.
