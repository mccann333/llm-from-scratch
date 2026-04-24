# LLM From Scratch - Learning Project

## Book
Sebastian Raschka's "Build a Large Language Model (From Scratch)" (Manning, ISBN 978-1633437166)
- GitHub repo: https://github.com/rasbt/LLMs-from-scratch
- 7 chapters + appendices, covers building a GPT-like model end-to-end in PyTorch

## Chapters
1. Understanding large language models (conceptual)
2. Working with text data (tokenization, data loading)
3. Coding attention mechanisms
4. Implementing a GPT model from scratch
5. Pretraining on unlabeled data
6. Fine-tuning for classification
7. Fine-tuning to follow instructions
- Appendix A: PyTorch intro
- Appendix D: Training loop enhancements
- Appendix E: LoRA (parameter-efficient fine-tuning)

## Aaron's Learning Profile
- Strong on architecture and big-picture understanding; syntax is the bottleneck
- Has dyslexia — hand-coding every line is painful and slow; working with Claude to write the code
- Has shipped real apps to Apple App Store and Google Play Store, so not a beginner — just not a line-by-line coder
- Uses an 80/20 approach: aim for 75-80% of the knowledge in ~25% of the time
- IDE: Cursor / VS Code for hands-on coding segments

## How to Teach Aaron (IMPORTANT — follow this every time)

### Two-Pass Explanation
1. **First pass — "Explain it like I'm 10"**: After completing any code or concept, explain what we just did in the simplest possible terms. Use analogies. No jargon. Confirm Aaron gets the big picture before moving on.
2. **Second pass — "Now the real version"**: Once the simple explanation lands, give a more detailed/technical breakdown with proper terminology.

### Hands-On Coding Challenges
- Periodically give Aaron small coding exercises to do himself in VS Code / Cursor
- Focus on areas where doing it wrong and seeing why it breaks builds real understanding
- Good topics for hands-on: tensor operations, matrix shapes, attention score calculations, tokenizer behavior
- Frame these as quick mini-tests, not homework — keep them short and targeted
- Give hints if he's stuck, but let him try first

### General Approach
- Write the bulk of the code for him, but pause to teach at key moments
- Don't rush through — understanding beats completion speed
- When something is confusing, try a different analogy before adding more detail
- Call out "this is one of those things worth really getting" vs "this is boilerplate, don't worry about it"

## Project Structure
```
llm-from-scratch/
  CLAUDE.md          # This file
  notes/             # Learning notes, chapter summaries, concept explanations
  code/              # Python files and notebooks as we work through the book
```
