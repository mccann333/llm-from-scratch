# Chapter 2 — `ch02_text_data.py` Explained (Like You're 10)

> Plain-English, block-by-block walkthrough of the Chapter 2 code.
> **Source file:** `code/ch02_text_data.py`

**The whole point of this file:** a computer can't read letters — it only understands numbers. So this file is the **translation station**: it takes a story written in words and turns it into neat rows of numbers the model can chew on. Then it bundles those numbers into bite-size practice chunks.

---

## 📖 Block 0 — Read the story (lines 1–8)

```python
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of characters:", len(raw_text))
print(raw_text[:99])
```

🧒 First you need something to read. This **opens a text file** (a short story called "The Verdict") and pours the entire thing into one big box called `raw_text`. The `"r"` means *"read"* (don't change it), and `encoding="utf-8"` is just *"use the normal alphabet+symbols rulebook"* so weird characters don't break.

🧒 Then it prints two quick sanity checks: *how many characters total* (so you know it actually loaded), and *the first 99 characters* (so you can eyeball that it's really the story).

> 💡 *In-between detail:* `with open(...) as f:` is the polite way to open a file — it automatically **closes the file** for you when it's done, like a door that shuts itself behind you.

---

## ✂️ Block 1 — Chop the text into pieces (lines 10–22)

```python
import re

text = "Hello, world. This, is a test."
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item for item in result if item.strip()]
print(result)

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])
```

🧒 You can't hand the model a giant blob of text — you have to **cut it into little pieces** first (each piece is called a **token**: usually a word or a punctuation mark). This block does the cutting.

🧒 `re` is the "pattern-matching" toolbox. `re.split(...)` means *"chop this text wherever you see a space or a punctuation mark."* That scary string of symbols is just a **list of all the places to cut**: commas, periods, question marks, dashes, spaces, and so on.

🧒 The second line — `[item ... if item.strip()]` — is a **clean-up sweep**: it throws away the empty leftover bits and blank spaces, keeping only the real pieces.

> 💡 *In-between detail:* It does this twice on purpose. The **first** time (`text = "Hello, world..."`) is a tiny *test sentence* to prove the chopper works. The **second** time runs it on the **whole story** (`raw_text`). `len(preprocessed)` tells you how many pieces you ended up with; `preprocessed[:30]` peeks at the first 30. (`.strip()` just means "trim the blank space off the edges of a piece.")

---

## 🔢 Block 2 — Build a dictionary, give every word a number (lines 24–33)

```python
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

vocab = {token: integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break
```

🧒 Now you make a **dictionary** — but not the kind that gives definitions. This one gives every unique word **its own ID number** (like assigning every kid in school a locker number).

🧒 Step by step:
- `set(preprocessed)` — throw out duplicates, so each word appears **once**.
- `sorted(...)` — put them in alphabetical order.
- `vocab_size = len(...)` — count how many different words there are.
- `vocab = {token: integer ...}` — pair each word with a number: `"a" → 0`, `"about" → 1`, and so on.

🧒 The little loop at the bottom just **prints the first 51 pairs** so you can see it worked (`break` means *"stop early once you've seen 51"* — you don't need to print thousands).

> 💡 *In-between detail:* `enumerate` is a handy helper that hands you **both** a counter *and* the item at the same time — that's how each word gets matched with a counting number.

---

## 🔁 Block 3 — `SimpleTokenizerV1`: the two-way translator (lines 36–51)

```python
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
```

🧒 This bundles the whole translation idea into one reusable **machine** with two buttons. The things it HAS (in `__init__`) are **two lookup tables**:
- `str_to_int` — word → number (for translating *into* number-language).
- `int_to_str` — number → word (for translating *back*).

🧒 The things it DOES:
- **`encode`** = *words → numbers.* It chops the text into pieces (same chopper as before), then looks up each piece's number.
- **`decode`** = *numbers → words.* It looks up each number's word and glues them back into a sentence. The `re.sub(...)` line is a **tidy-up**: it removes the awkward space before punctuation, so you get `"Hello, world."` instead of `"Hello , world ."`.

> 💡 *In-between detail:* `encode` and `decode` are mirror images — one goes in, the other comes back out. If you encode a sentence and then decode it, you should get your sentence back. (`" ".join([...])` just means "stick all these word-pieces together with a space between each.")

---

## 🚫 Block 4 — Special tokens for "the end" and "huh?" (lines 54–92)

```python
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}
print(len(vocab))
...

class SimpleTokenizerV2:
    ...
    def encode(self, text):
        ...
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]
        ...
```

🧒 The V1 translator had a problem: if you gave it a word it had **never seen before**, it would crash — like a translator freezing on a word that's not in their book. This block fixes that by adding **two special words** to the dictionary:
- `<|unk|>` = *"unknown — a word I don't recognize."*
- `<|endoftext|>` = *"this is where one document ends and another begins."*

🧒 `SimpleTokenizerV2` is the **upgraded** translator. It's identical to V1 except for one smart line in `encode`: *"if I know this word, use it; otherwise, replace it with `<|unk|>`."* So it never crashes on a surprise word — it just shrugs and says "unknown."

🧒 The test at the bottom glues two sentences together with `<|endoftext|>` in the middle, encodes it, then decodes it — proving the upgraded machine handles both special words.

> 💡 *In-between detail:* `.extend([...])` means *"add these extra items onto the end of the list."* This is the model learning to say "I don't know that one" instead of breaking — a small but important real-world survival skill.

---

## 🪟 Block 5 — A real-world tokenizer + the sliding window (lines 95–117)

```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
...
enc_text = tokenizer.encode(raw_text)
...
enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]
print("x:", x)
print("y:", y)

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), '---->',  tokenizer.decode([desired]))
```

🧒 Two big ideas here.

**First — a grown-up tokenizer.** Your homemade translator works, but the real GPT-2 uses a fancier one called **`tiktoken`** that splits words into smart sub-pieces (so "playing" might become "play" + "ing"). This is the *exact same tokenizer* you use over in ch04. From here on, you use the professional tool.

**Second — the core training trick (`x` and `y`):** how do you teach a model to predict the next word? You show it a few words (`x`) and tell it the **very next word** (`y`) — which is just `x` shifted over by one. That's the whole game:
- `x` = `[word1, word2, word3, word4]`
- `y` = `[word2, word3, word4, word5]`

🧒 The little loop makes this crystal clear by printing it like flashcards:
> `"I had"` ----> `"always"`
> `"I had always"` ----> `"thought"`

Each line is one practice question: *given the words on the left, guess the word on the right.* That's literally how the model learns to write.

> 💡 *In-between detail:* `enc_sample = enc_text[50:]` just means *"skip the first 50 tokens and start from there"* — no special reason, just grabbing a sample from the middle. The "sliding window" name comes from how `x` slides forward one word at a time to make the next question.

---

## 📦 Block 6 — `GPTDatasetV1` + `create_dataloader_v1`: the practice-chunk factory (lines 120–169)

```python
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            drop_last=drop_last, num_workers=num_workers)
    return dataloader
```

🧒 Doing that "x and y" flashcard trick by hand for a whole book would take forever. This block builds a **factory** that stamps out *thousands* of flashcards automatically.

🧒 **`GPTDatasetV1`** is a tidy box that HOLDS all the flashcards. Its `__init__` runs the slider across the whole story: every few words it snips out an `input_chunk` (the question) and a `target_chunk` (the answer, shifted by one), and stores them.
- `max_length` = how many words are on each flashcard.
- `stride` = how far the window jumps before snipping the next card.

🧒 The two little DOES at the bottom are special helpers PyTorch looks for: **`__len__`** answers *"how many flashcards do you have?"* and **`__getitem__`** answers *"give me flashcard number 5."*

🧒 **`create_dataloader_v1`** wraps it all in a bow. A `DataLoader` is a **conveyor belt** that feeds flashcards to the model in **batches** (stacks of several at once, which is faster), and can **shuffle** them so the model doesn't just memorize the order.

> 💡 *In-between detail:* `torch.tensor(...)` turns a plain list of numbers into a **tensor** — the special grid-of-numbers format PyTorch needs (think: a spreadsheet the math toolbox can crunch super fast). The `batch_size=4, max_length=256` bits are **default settings** — used unless you say otherwise.

---

## 🃏 Block 7 — Token embeddings: turning IDs into meaning (lines 172–197)

```python
input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)

# --- Encoding word positions ---
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print(inputs)
print(inputs.shape)
```

🧒 Here's the last big idea. A word's ID number (like `2`) is just a **name tag** — it doesn't actually mean anything. `2` isn't "bigger" or "better" than `1`. So we need to turn each ID into something with real **meaning**.

🧒 **`nn.Embedding`** is a big **lookup table of meaning cards.** You give it an ID number, it hands back a list of numbers (the word's "meaning"). Those numbers start out random, but during training they get nudged until similar words end up with similar cards (so "king" and "queen" sit near each other).
- `vocab_size` = how many cards (one per word in the dictionary).
- `output_dim` = how many numbers on each card.

🧒 The bottom part grabs a real batch of `inputs` from your conveyor belt and prints `inputs.shape` — checking the grid is the size you expect (8 flashcards × 4 words each).

> 💡 *In-between detail:* This is the **exact same `nn.Embedding`** that becomes `tok_emb` over in ch04 — you've seen it before! And `torch.manual_seed(123)` just means *"start the randomness from the same spot every time,"* so you get the same numbers on each run and nothing feels mysteriously different. (The chapter also adds **position cards** next to the meaning cards, so the model knows word *order* — that became `pos_emb` in ch04.)

---

**ch02 end to end:** story → pieces → ID numbers → meaning cards → practice flashcards on a conveyor belt. Everything downstream eats what this file produces.
