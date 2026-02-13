# nanochat Wiki Catalogue

**Generated:** 2024
**Repository:** https://github.com/karpathy/nanochat (branch: master)
**Location:** `/Users/govindm4max/code/tmp/oss/gh/nanochat/wiki/catalogue.json`

---

## Overview

This hierarchical wiki catalogue provides a comprehensive documentation structure for the nanochat repository — a minimal experimental harness for training Large Language Models (LLMs) on a single GPU node.

### Catalogue Statistics

- **Total Sections:** 2 top-level sections
- **Total Pages:** 23 documentation pages
- **Total Prompts:** 160 prompts with source code citations
- **Files Referenced:** 32 unique source files
- **Nesting Depth:** Maximum 4 levels
- **Citation Format:** `[file_path:line_number](https://github.com/karpathy/nanochat/blob/master/file_path#Lline_number)`

---

## Structure

### 1. Getting Started (4 pages, 24 prompts)

Quick introduction to nanochat, environment setup, and basic training workflows.

- **Overview** — What is nanochat and why it exists
- **Installation & Environment Setup** — Dependencies, GPU requirements, configuration
- **GPT-2 Speedrun Walkthrough** — Complete pipeline to train and chat with GPT-2
- **Quick Reference** — Common commands, key hyperparameters, troubleshooting

### 2. Deep Dive (6 subsections, 19 pages, 136 prompts)

Technical deep-dives into architecture, implementation, training stages, and optimization.

#### 2.1 Architecture (4 pages, 25 prompts)
- **GPT Transformer Model** — RoPE, QK norm, ReLU², GQA, sliding window attention
- **Attention Mechanism** — Flash Attention 3, RoPE, KV cache
- **MLP Blocks & ResFormer** — ReLU² activation, value embeddings, residuals
- **Model Scaling Laws** — Automatic hyperparameter calculation, Chinchilla scaling

#### 2.2 Data Pipeline (3 pages, 19 prompts)
- **BPE Tokenizer** — GPT-4-style tokenizer with rustbpe and tiktoken
- **Dataset Preparation** — FineWeb-Edu preprocessing, parquet conversion
- **Distributed Dataloader** — BOS-aligned bestfit packing, DDP sharding

#### 2.3 Training Stages (3 pages, 28 prompts)
- **Base Model Pretraining** — Unsupervised training with mixed Muon/AdamW
- **Supervised Fine-Tuning (SFT)** — Task mixture training with chat formatting
- **Reinforcement Learning (RL)** — Simplified GRPO/REINFORCE on GSM8K

#### 2.4 Optimization (3 pages, 23 prompts)
- **Mixed Muon/AdamW Optimizer** — Combined optimizer with Polar Express
- **Learning Rate Schedule** — Warmup, warmdown, cosine annealing
- **FP8 Training** — Float8 mixed precision with tensorwise scaling

#### 2.5 Evaluation (3 pages, 22 prompts)
- **DCLM CORE Metric** — Primary evaluation metric across multiple tasks
- **Individual Task Evaluations** — GSM8K, MMLU, ARC, HumanEval, SmolTalk, SpellingBee
- **Bits Per Byte (BPB) Metric** — Vocab-size-invariant loss metric

#### 2.6 Inference & Deployment (3 pages, 19 prompts)
- **Inference Engine** — KV cache engine with tool use support
- **Chat Interfaces (CLI & Web)** — Terminal and web-based chat UIs
- **Data-Parallel Serving** — Multi-GPU inference with worker pools

---

## Key Features

### Citation-Backed Documentation
Every prompt in the catalogue references specific source files with line numbers, ensuring all documentation is traceable to actual implementation:
- Example: `"Rotary embeddings (RoPE): apply_rotary_emb splits hidden dim in half and rotates pairs of dimensions (nanochat/gpt.py:51-57)"`

### Progressive Learning Path
The catalogue is designed for progressive discovery:
1. **Getting Started** — Onboard new users quickly with practical examples
2. **Deep Dive** — Technical depth for researchers and contributors

### Actual Implementation Focus
All section titles and prompts are derived from the actual codebase structure, not generic templates. For example:
- "Mixed Muon/AdamW Optimizer" (actual implementation in `nanochat/optim.py`)
- "BOS-aligned bestfit packing" (actual algorithm in `nanochat/dataloader.py`)
- "Flash Attention 3 integration" (actual feature in `nanochat/flash_attention.py`)

### Compute-Optimal Focus
The catalogue emphasizes nanochat's unique design principle: the `--depth` dial that automatically configures all hyperparameters for compute-optimal training at various model scales.

---

## Usage

This catalogue serves as the foundation for generating comprehensive wiki pages using the wiki-page-writer agent. Each prompt can be expanded into a full documentation page with:
- Mermaid diagrams (architecture, sequence, class diagrams)
- Detailed explanations with source code citations
- Code examples from the actual implementation
- Cross-references to related pages

### Example Page Generation

For the "GPT Transformer Model" page (23 prompts → full documentation page):
```bash
# Use the wiki-page-writer skill with prompts from catalogue
wiki-page-writer --page-id "gpt-model" --prompts-from catalogue.json
```

---

## Validation

All 160 prompts have been validated:
- ✅ All 32 referenced files exist in the repository
- ✅ All citations use correct file paths
- ✅ Maximum nesting depth: 4 levels (within constraint)
- ✅ Maximum children per section: 6-8 (within constraint)

---

## Next Steps

1. **Generate Full Wiki Pages:** Use the catalogue prompts to generate complete documentation pages with the wiki-page-writer agent
2. **Add Onboarding Guides:** Generate 4 audience-tailored guides (Contributor, Staff Engineer, Executive, Product Manager)
3. **Package as VitePress Site:** Build a browsable static site with dark theme and Mermaid diagrams
4. **Generate llms.txt:** Create LLM-friendly summaries following the llms.txt specification

---

## Repository Context

**nanochat** is the simplest experimental harness for training LLMs by Andrej Karpathy. It covers all major LLM stages on a single GPU node:
- Tokenization (BPE with rustbpe/tiktoken)
- Pretraining (FineWeb-Edu, mixed Muon/AdamW, FP8 support)
- Supervised Fine-Tuning (GSM8K, MMLU, SmolTalk, SpellingBee)
- Reinforcement Learning (simplified GRPO/REINFORCE)
- Evaluation (DCLM CORE metric, bits per byte)
- Inference (KV cache engine, tool use, chat UI)

**Key Achievement:** Train GPT-2 capability (0.256 DCLM CORE score) in ~2.76 hours on 8XH100 for ~$72, compared to the original $43,000 cost in 2019.
