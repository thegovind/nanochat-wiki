---
layout: home
hero:
  name: nanochat Wiki
  text: The minimal full-stack ChatGPT clone
  tagline: Train your own GPT-2 for ~$72 in ~3 hours. Comprehensive documentation for Andrej Karpathy's nanochat.
  image:
    src: /logo.svg
    alt: nanochat
  actions:
    - theme: brand
      text: Get Started →
      link: /01-getting-started/overview
    - theme: alt
      text: Onboarding Guide
      link: /onboarding-guide
features:
  - icon: 🧠
    title: Full-Stack LLM Training
    details: Tokenization → Pretraining → SFT → RL → Evaluation → Inference → Chat UI, all in one repo
  - icon: 🎛️
    title: One Dial - depth
    details: Set --depth and all hyperparameters are auto-calculated for compute-optimal training
  - icon: ⚡
    title: GPU Optimized
    details: Flash Attention 3, FP8 training, Muon optimizer, sliding window attention, and distributed training
  - icon: 🔧
    title: Minimal & Hackable
    details: No giant config objects or framework abstractions — clean, readable PyTorch code designed for forking
---

## Welcome to nanochat

**nanochat** is [Andrej Karpathy](https://github.com/karpathy)'s minimal full-stack ChatGPT clone — a single repository that takes you from raw text all the way to a working chat interface. Every stage of the LLM pipeline is implemented in clean, readable PyTorch with a single `--depth` dial that auto-configures all hyperparameters for compute-optimal training. A GPT-2 grade model can be trained from scratch for approximately **$72 in ~3 hours on 8×H100 GPUs**.

### Architecture Overview

The diagram below shows the end-to-end pipeline that nanochat implements:

```mermaid
flowchart LR
    A["📚 Dataset"] --> B["🔤 Tokenizer"]
    B --> C["📦 Dataloader"]
    C --> D["🧠 GPT Model"]
    D --> E["⚙️ Pretraining"]
    E --> F["🎯 SFT"]
    F --> G["🏆 RL"]
    G --> H["📊 Evaluation"]
    H --> I["🚀 Inference Engine"]
    I --> J["💬 Web UI"]

    style A fill:#2d333b,stroke:#58a6ff,color:#c9d1d9
    style B fill:#2d333b,stroke:#58a6ff,color:#c9d1d9
    style C fill:#2d333b,stroke:#58a6ff,color:#c9d1d9
    style D fill:#2d333b,stroke:#f78166,color:#c9d1d9
    style E fill:#2d333b,stroke:#f78166,color:#c9d1d9
    style F fill:#2d333b,stroke:#f78166,color:#c9d1d9
    style G fill:#2d333b,stroke:#f78166,color:#c9d1d9
    style H fill:#2d333b,stroke:#3fb950,color:#c9d1d9
    style I fill:#2d333b,stroke:#3fb950,color:#c9d1d9
    style J fill:#2d333b,stroke:#3fb950,color:#c9d1d9
```

### Key Module Map

```mermaid
graph TD
    subgraph Data["Data Layer"]
        DS["nanochat/dataset.py<br>Parquet download"]
        TK["nanochat/tokenizer.py<br>BPE tokenizer"]
        DL["nanochat/dataloader.py<br>Bestfit packing"]
    end

    subgraph Model["Model Layer"]
        GPT["nanochat/gpt.py<br>GPT + RoPE + GQA"]
        FA["nanochat/flash_attention.py<br>FA3 / SDPA"]
        FP8["nanochat/fp8.py<br>FP8 matmul"]
    end

    subgraph Training["Training Layer"]
        BT["scripts/base_train.py<br>Pretraining loop"]
        SFT["scripts/chat_sft.py<br>Supervised finetuning"]
        RL["scripts/chat_rl.py<br>REINFORCE RL"]
        OPT["nanochat/optim.py<br>Muon + AdamW"]
    end

    subgraph Inference["Inference Layer"]
        ENG["nanochat/engine.py<br>KV cache + generation"]
        WEB["scripts/chat_web.py<br>FastAPI server"]
        EXEC["nanochat/execution.py<br>Sandboxed code exec"]
    end

    DS --> TK --> DL --> BT
    GPT --> BT
    GPT --> SFT
    GPT --> RL
    OPT --> BT
    OPT --> SFT
    OPT --> RL
    FA --> GPT
    FP8 --> GPT
    BT --> SFT --> RL
    GPT --> ENG --> WEB
    EXEC --> ENG

    style Data fill:#161b22,stroke:#30363d,color:#e6edf3
    style Model fill:#161b22,stroke:#30363d,color:#e6edf3
    style Training fill:#161b22,stroke:#30363d,color:#e6edf3
    style Inference fill:#161b22,stroke:#30363d,color:#e6edf3
    style DS fill:#2d333b,stroke:#58a6ff,color:#e6edf3
    style TK fill:#2d333b,stroke:#58a6ff,color:#e6edf3
    style DL fill:#2d333b,stroke:#58a6ff,color:#e6edf3
    style GPT fill:#2d333b,stroke:#f78166,color:#e6edf3
    style FA fill:#2d333b,stroke:#f78166,color:#e6edf3
    style FP8 fill:#2d333b,stroke:#f78166,color:#e6edf3
    style BT fill:#2d333b,stroke:#6d5dfc,color:#e6edf3
    style SFT fill:#2d333b,stroke:#6d5dfc,color:#e6edf3
    style RL fill:#2d333b,stroke:#6d5dfc,color:#e6edf3
    style OPT fill:#2d333b,stroke:#6d5dfc,color:#e6edf3
    style ENG fill:#2d333b,stroke:#3fb950,color:#e6edf3
    style WEB fill:#2d333b,stroke:#3fb950,color:#e6edf3
    style EXEC fill:#2d333b,stroke:#3fb950,color:#e6edf3
```
