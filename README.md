# Do Vision-Language Models Perform Causal Reasoning or Shortcut Learning?

Evaluated BLIP-2 (OPT-2.7B) on a custom-designed diagnostic dataset to test whether multimodal models genuinely reason from visual evidence or exploit statistical shortcuts in language. Built a controlled 150-sample benchmark across three reasoning conditions and ran zero-shot inference to expose failure modes.

---

## The Problem

Most vision-language models report strong benchmark numbers — but benchmarks don't tell you *why* a model gets an answer right. A model could answer "Is there a cat in this image?" correctly because it saw the cat, or because the word "cat" in the question biased it toward "yes." These two explanations have completely different implications for how trustworthy the model is.

This project investigates: **does BLIP-2 reason from images, or from language priors?**

---

## Approach

Instead of training or fine-tuning anything, this is a pure **diagnostic evaluation** — inference only. The core idea is to design questions where a model relying on shortcuts will fail, and a model doing real visual reasoning will succeed.

Three dataset categories were constructed:

| Category | Design | What it tests |
|---|---|---|
| **Normal** | Image and question are straightforwardly aligned | Baseline capability |
| **Spurious** | Question embeds a linguistic prior ("always", "typically") that contradicts the image | Whether model uses language shortcuts over visual evidence |
| **Counterfactual** | Question directly contradicts real-world expectations | Whether model reasons from the image or from memorized patterns |

**Example — Spurious sample:**
> Image: a motorcycle parked with no rider
> Question: *"Motorcycles are always ridden by someone. Is there a rider on this motorcycle?"*
> Correct answer: **no**
> BLIP-2 answer: **yes** ← fell for the linguistic prior

**Example — Counterfactual sample:**
> Image: a train on a hill
> Question: *"Is this train flying through the sky?"*
> Correct answer: **no**
> BLIP-2 answer: **no** ← correctly grounded in the image

---

## Dataset

- **150 samples total** — 50 normal, 50 spurious, 50 counterfactual
- Images sourced from **COCO val2017** (verified locally, zero broken links)
- 23 unique images, each used across multiple categories with different questions
- All questions are yes/no format for clean, unambiguous evaluation
- Spurious questions deliberately embed factual world-priors using trigger words like *"always"*, *"typically"*, *"are known for"*

Image descriptions generated using **BLIP-base** to ensure ground-truth descriptions matched actual image content before writing questions.

---

## Model

| Setting | Value |
|---|---|
| Model | BLIP-2 OPT-2.7B |
| Source | `Salesforce/blip2-opt-2.7b` via HuggingFace |
| Inference | Zero-shot, no fine-tuning |
| Precision | float16 |
| Hardware | Google Colab T4 GPU |
| Decoding | Greedy, max 10 new tokens |
| Input format | `"Question: {question} Answer:"` |

---

## Results

| Category | Correct | Accuracy |
|---|---|---|
| Normal | 45 / 50 | **90.0%** |
| Spurious | 13 / 50 | **26.0%** |
| Counterfactual | 39 / 50 | **78.0%** |
| **Overall** | **97 / 150** | **64.7%** |

---

## Failure Pattern Analysis

| Category | Fell for linguistic prior (NO→YES) | Missed visual evidence (YES→NO) | Other |
|---|---|---|---|
| Normal | 0 | 4 | 1 |
| Spurious | **36** | 0 | 1 |
| Counterfactual | 11 | 0 | 0 |

The pattern is clean and telling. On normal questions the model fails only because it misses something visually. On spurious questions it fails almost exclusively by saying YES when the answer is NO — the exact signature of a model reading the question's implied prior rather than looking at the image.

---

## Key Finding

**Linguistic prior trap rate: 36/50 = 72%**

When a spurious question implied something was "always" true, BLIP-2 agreed 72% of the time — even when the image directly contradicted it. In multiple cases the model hallucinated objects that were not present:

- *"yes, there are towels visible"* — no towels in the image
- *"yes, there are people using these toilets"* — no people present  
- *"yes, there are birds visible near this river"* — no birds in frame
- *"yes, but it is not the sheepdog's"* — no sheepdog, model invented a narrative

The 64-point accuracy drop from normal (90%) to spurious (26%) is entirely explained by the model treating the question's factual premise as ground truth rather than looking at the image.

The counterfactual results (78%) tell a complementary story — when questions are absurd enough ("is the train underwater?"), the model resists. The failure zone is specifically linguistic priors that are *plausible* — claims that sound factually reasonable even when they don't match the image.

---

## Conclusion

BLIP-2 shows strong evidence of shortcut learning under controlled diagnostic conditions. A 64-point accuracy drop on spurious questions, combined with a 72% linguistic prior trap rate, confirms the model frequently substitutes language-pattern matching for genuine visual reasoning.

This has direct implications for how VLMs are evaluated and deployed. Standard benchmarks don't distinguish between a model that reasons and a model that pattern-matches — this project shows the gap is real and measurable.

---


## Repository Structure
```
Multimodal-Shortcut-Eval/
├── multimodal_experiment.ipynb     ← full pipeline: data → inference → analysis
├── eval_dataset_v5_final.json      ← 150-sample benchmark dataset
├── inference_results.csv           ← raw predictions for all 150 samples
└── README.md
```

---

## Dependencies
```
transformers
accelerate  
bitsandbytes
pillow
torch
pandas
matplotlib
```
