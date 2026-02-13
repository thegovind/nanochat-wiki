---
title: nanochat Wiki
description: Technical documentation for nanochat — Andrej Karpathy's minimal LLM training codebase
layout: home
hero:
  name: nanochat
  text: Train GPT-2 from scratch in ~3 hours
  tagline: "~2500 lines of PyTorch · 8×H100 · ~$72 · no frameworks"
  actions:
    - theme: brand
      text: Read the Docs →
      link: /getting-started/overview
    - theme: alt
      text: Source Code
      link: https://github.com/karpathy/nanochat
---

<div class="home-content">

## Quick Start

```bash
git clone https://github.com/karpathy/nanochat.git && cd nanochat
pip install -r requirements.txt
python train.py --depth 1    # single complexity dial controls everything
python sft.py                # supervised fine-tuning
python rl.py                 # GRPO on GSM8K
python chat.py               # launch the chat UI
```

## What's Under the Hood

| Component | Implementation | Docs |
|-----------|---------------|------|
| **Model** | GPT-2 w/ RoPE, GQA, ReLU², sliding window attention | [Architecture →](./deep-dive/architecture/gpt-model) |
| **Attention** | Flash Attention 3 + SDPA fallback, KV cache | [Attention →](./deep-dive/architecture/attention-mechanisms) |
| **Tokenizer** | BPE via RustBPE + tiktoken (100K vocab) | [Tokenizer →](./deep-dive/data-pipeline/tokenizer) |
| **Data** | FineWeb-Edu 100BT, BOS-aligned bestfit packing | [Dataloader →](./deep-dive/data-pipeline/dataloader) |
| **Optimizer** | Fused Muon + AdamW, per-layer LR scaling | [Optimizer →](./deep-dive/optimization/muon-adamw) |
| **FP8** | Float8Linear with dynamic scaling | [FP8 →](./deep-dive/optimization/fp8-training) |
| **Training** | DDP/FSDP pretraining → SFT → GRPO | [Pretraining →](./deep-dive/training/pretraining) |
| **Eval** | CORE metric, GSM8K, MMLU, ARC, HumanEval | [Evaluation →](./deep-dive/evaluation/core-metric) |
| **Inference** | KV cache engine, FastAPI server, sandboxed tool use | [Engine →](./deep-dive/inference/inference-engine) |

## Architecture at a Glance

```
train.py          →  pretraining (DDP, 8×H100, FP8, ~3hr)
  └─ model.py     →  GPT: RoPE + GQA + ReLU² + sliding window
  └─ data.py      →  FineWeb-Edu 100BT shards, bestfit packing
  └─ optim.py     →  Muon (matrix params) + AdamW (vectors/embeds)
sft.py            →  supervised fine-tuning on multi-task data
rl.py             →  GRPO reinforcement learning on GSM8K
chat.py           →  FastAPI server + vanilla JS chat UI
  └─ inference.py →  KV cache, batched generation
  └─ sandbox.py   →  sandboxed Python code execution
eval.py           →  CORE metric, BPB, task evals
```

## Browse the Docs

**Getting Started** — [Overview](./getting-started/overview) · [Installation](./getting-started/installation) · [Speedrun Walkthrough](./getting-started/speedrun-walkthrough) · [Quick Reference](./getting-started/quick-reference)

**Deep Dive** — [GPT Model](./deep-dive/architecture/gpt-model) · [Attention](./deep-dive/architecture/attention-mechanisms) · [MLP & Blocks](./deep-dive/architecture/mlp-and-blocks) · [Scaling](./deep-dive/architecture/scaling-and-configuration) · [Tokenizer](./deep-dive/data-pipeline/tokenizer) · [Dataset](./deep-dive/data-pipeline/dataset) · [Dataloader](./deep-dive/data-pipeline/dataloader) · [Pretraining](./deep-dive/training/pretraining) · [SFT](./deep-dive/training/supervised-finetuning) · [RL](./deep-dive/training/reinforcement-learning) · [Optimizer](./deep-dive/optimization/muon-adamw) · [LR Schedule](./deep-dive/optimization/learning-rate-schedule) · [FP8](./deep-dive/optimization/fp8-training) · [CORE Metric](./deep-dive/evaluation/core-metric) · [Eval Tasks](./deep-dive/evaluation/evaluation-tasks) · [BPB](./deep-dive/evaluation/bits-per-byte) · [Inference Engine](./deep-dive/inference/inference-engine) · [Chat UI](./deep-dive/inference/chat-ui) · [Code Execution](./deep-dive/inference/code-execution)

**Onboarding** — [Contributor](./onboarding/contributor) · [Staff Engineer](./onboarding/staff-engineer) · [Executive](./onboarding/executive) · [Product Manager](./onboarding/product-manager)

</div>
