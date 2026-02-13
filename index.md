---
title: nanochat Documentation
description: Comprehensive documentation for nanochat — the simplest experimental harness for training LLMs
layout: home
hero:
  name: nanochat
  text: The simplest experimental harness for training LLMs
  tagline: Train your own GPT-2 capability model for ~$72 in ~3 hours
  actions:
    - theme: brand
      text: Get Started
      link: /getting-started/overview
    - theme: alt
      text: Deep Dive
      link: /deep-dive/architecture/gpt-model
    - theme: alt
      text: GitHub
      link: https://github.com/karpathy/nanochat
features:
  - icon: 🚀
    title: Full-Stack LLM Training
    details: Covers tokenization, pretraining, SFT, RL, evaluation, inference, and chat UI
  - icon: 🎯
    title: Single Complexity Dial
    details: One --depth parameter auto-configures all hyperparameters for compute-optimal training
  - icon: 💰
    title: GPT-2 for $72
    details: Train GPT-2 capability models in ~3 hours on 8XH100 GPUs — down from $43K in 2019
  - icon: 🔧
    title: Minimal & Hackable
    details: No framework bloat, no config objects — just clean, readable, forkable PyTorch code
---

## 📖 Documentation Guide

### Getting Started
| Page | Description |
|------|-------------|
| [Overview](./getting-started/overview) | What is nanochat and why it exists |
| [Installation](./getting-started/installation) | Environment setup and hardware requirements |
| [GPT-2 Speedrun](./getting-started/speedrun-walkthrough) | Complete pipeline walkthrough |
| [Quick Reference](./getting-started/quick-reference) | Commands, metrics, and troubleshooting |

### Deep Dive

#### Architecture
| Page | Description |
|------|-------------|
| [GPT Transformer Model](./deep-dive/architecture/gpt-model) | RoPE, QK norm, ReLU², GQA, sliding windows |
| [Attention Mechanisms](./deep-dive/architecture/attention-mechanisms) | Flash Attention 3, SDPA fallback, KV cache |
| [MLP & Blocks](./deep-dive/architecture/mlp-and-blocks) | Residual connections, per-layer scalars |
| [Scaling & Configuration](./deep-dive/architecture/scaling-and-configuration) | Depth dial, FLOPS estimation |

#### Data Pipeline
| Page | Description |
|------|-------------|
| [Tokenizer](./deep-dive/data-pipeline/tokenizer) | BPE with RustBPE + tiktoken |
| [Dataset](./deep-dive/data-pipeline/dataset) | FineWeb-Edu 100BT preparation |
| [Dataloader](./deep-dive/data-pipeline/dataloader) | BOS-aligned bestfit packing |

#### Training Stages
| Page | Description |
|------|-------------|
| [Pretraining](./deep-dive/training/pretraining) | Distributed base model training |
| [Supervised Fine-Tuning](./deep-dive/training/supervised-finetuning) | Multi-task instruction tuning |
| [Reinforcement Learning](./deep-dive/training/reinforcement-learning) | Simplified GRPO on GSM8K |

#### Optimization
| Page | Description |
|------|-------------|
| [Muon/AdamW Optimizer](./deep-dive/optimization/muon-adamw) | Fused mixed optimizer |
| [Learning Rate Schedule](./deep-dive/optimization/learning-rate-schedule) | Warmup, warmdown, scaling |
| [FP8 Training](./deep-dive/optimization/fp8-training) | Float8Linear, dynamic scaling |

#### Evaluation
| Page | Description |
|------|-------------|
| [CORE Metric](./deep-dive/evaluation/core-metric) | DCLM CORE evaluation |
| [Evaluation Tasks](./deep-dive/evaluation/evaluation-tasks) | GSM8K, MMLU, ARC, HumanEval |
| [Bits Per Byte](./deep-dive/evaluation/bits-per-byte) | Vocab-independent loss metric |

#### Inference & Deployment
| Page | Description |
|------|-------------|
| [Inference Engine](./deep-dive/inference/inference-engine) | KV cache, batched generation |
| [Chat UI](./deep-dive/inference/chat-ui) | FastAPI web server |
| [Code Execution](./deep-dive/inference/code-execution) | Sandboxed Python execution |

### Onboarding Guides
| Guide | Audience |
|-------|----------|
| [Contributor Guide](./onboarding/contributor) | New contributors with Python/PyTorch experience |
| [Staff Engineer Guide](./onboarding/staff-engineer) | Senior engineers evaluating architecture |
| [Executive Guide](./onboarding/executive) | VP/director-level technology leaders |
| [Product Manager Guide](./onboarding/product-manager) | PMs and non-engineering stakeholders |
