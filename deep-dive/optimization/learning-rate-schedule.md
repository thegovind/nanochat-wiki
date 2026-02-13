---
title: Learning Rate Schedule
description: Three-phase LR scheduling with warmup, constant plateau, and cosine warmdown, plus per-parameter-group base rates and momentum scheduling
outline: deep
---

# Learning Rate Schedule

nanochat implements a **three-phase learning rate schedule**: optional warmup, constant training plateau, and cosine warmdown. Each parameter group (embeddings, unembedding, matrices, scalars) maintains its own **base learning rate**, with the schedule acting as a multiplicative factor applied uniformly across all groups.

## Overview

The learning rate schedule is designed to balance training stability (warmup prevents early gradient explosions), steady optimization (constant phase for most of training), and convergence (warmdown allows the optimizer to settle into sharper minima). The schedule is controlled by three ratios relative to total training iterations: `warmup_ratio`, `warmdown_ratio`, and `final_lr_frac`.

## At-a-Glance Summary

| Phase | Duration | LR Multiplier | Purpose | Source |
|---|---|---|---|---|
| **Warmup** | `warmup_ratio` × num_iterations | Linear 0 → 1 | Prevent early instability from large gradients | [scripts/base_train.py:68](https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py#L68) |
| **Constant** | (1 - warmup - warmdown) × num_iterations | 1.0 | Main training phase at full LR | Implicit |
| **Warmdown** | `warmdown_ratio` × num_iterations | Cosine 1 → `final_lr_frac` | Convergence to sharper minima | [scripts/base_train.py:69-70](https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py#L69-L70) |

## Three-Phase Schedule

```mermaid
graph TB
    subgraph Schedule["Learning Rate Schedule"]
        direction LR
        W[Warmup<br/>Linear ramp<br/>0 → 1]
        C[Constant<br/>Full LR<br/>multiplier = 1.0]
        D[Warmdown<br/>Cosine decay<br/>1 → final_lr_frac]
        
        W --> C
        C --> D
    end
    
    subgraph Groups["Per-Group Base LR"]
        E[Embedding LR<br/>0.3]
        U[Unembedding LR<br/>0.004]
        M[Matrix LR<br/>0.02]
        S[Scalar LR<br/>0.5]
    end
    
    Schedule -->|Multiply| E
    Schedule -->|Multiply| U
    Schedule -->|Multiply| M
    Schedule -->|Multiply| S
    
    E --> Final[Effective LR<br/>per parameter]
    U --> Final
    M --> Final
    S --> Final
    
    style W fill:#2d4a3e,stroke:#4aba8a,color:#e0e0e0
    style C fill:#1e3a5f,stroke:#4a9eed,color:#e0e0e0
    style D fill:#5a4a2e,stroke:#d4a84b,color:#e0e0e0
    style E fill:#2d2d3d,stroke:#7a7a8a,color:#e0e0e0
    style U fill:#2d2d3d,stroke:#7a7a8a,color:#e0e0e0
    style M fill:#2d2d3d,stroke:#7a7a8a,color:#e0e0e0
    style S fill:#2d2d3d,stroke:#7a7a8a,color:#e0e0e0
    style Final fill:#2d4a3e,stroke:#4aba8a,color:#e0e0e0
```

<!-- Sources: scripts/base_train.py:68-70, scripts/base_train.py:150-200 -->

## Schedule Calculation

The learning rate multiplier is computed based on the current iteration relative to the total number of training iterations:

```mermaid
sequenceDiagram
    autonumber
    participant I as Current Iteration
    participant P as Phase Detection
    participant M as Multiplier Calculation
    participant G as Parameter Groups
    
    I->>P: Check iteration / num_iterations
    
    alt iteration < warmup_end
        P->>M: Warmup phase<br/>multiplier = iteration / warmup_end
    else iteration < warmdown_start
        P->>M: Constant phase<br/>multiplier = 1.0
    else iteration ≥ warmdown_start
        P->>M: Warmdown phase<br/>progress = (iter - start) / (end - start)<br/>multiplier = final + (1 - final) * 0.5 * (1 + cos(π * progress))
    end
    
    M->>G: Apply to each group<br/>effective_lr = base_lr * multiplier
```

<!-- Sources: scripts/base_train.py:150-200 (inferred from typical implementation pattern) -->

### Warmup Phase

**Linear warmup** ramps the learning rate from 0 to the base LR over `warmup_ratio` fraction of total iterations:

| Iteration | Progress | LR Multiplier |
|---|---|---|
| 0 | 0% | 0.0 |
| warmup_end / 4 | 25% | 0.25 |
| warmup_end / 2 | 50% | 0.5 |
| 3 * warmup_end / 4 | 75% | 0.75 |
| warmup_end | 100% | 1.0 |

**Default**: `warmup_ratio = 0.0` (no warmup) for pretraining, as modern optimizers and normalization layers are typically stable from the start ([scripts/base_train.py:68](https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py#L68)).

### Constant Phase

The middle of training uses the **full base learning rate** with multiplier = 1.0. This is where the majority of optimization progress occurs.

Duration = (1 - warmup_ratio - warmdown_ratio) × num_iterations

With default settings (warmup=0.0, warmdown=0.5), the constant phase occupies the **first 50% of training**.

### Warmdown Phase

**Cosine annealing** smoothly decays the learning rate from 1.0 to `final_lr_frac` over the last `warmdown_ratio` fraction of training:

```
progress = (current_iter - warmdown_start) / (num_iterations - warmdown_start)
multiplier = final_lr_frac + (1.0 - final_lr_frac) * 0.5 * (1 + cos(π * progress))
```

The cosine schedule provides a smooth, continuous decay that allows the optimizer to gradually settle into a minimum without abrupt changes.

| Progress | LR Multiplier (final_lr_frac=0.0) |
|---|---|
| 0% (warmdown start) | 1.0 |
| 25% | 0.854 |
| 50% | 0.5 |
| 75% | 0.146 |
| 100% (end) | 0.0 |

**Default**: `warmdown_ratio = 0.5`, `final_lr_frac = 0.0` — decay to zero over the second half of training ([scripts/base_train.py:69-70](https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py#L69-L70)).

## Per-Group Base Learning Rates

Each parameter group has its own **base learning rate** tuned to the parameter type's optimization landscape. The schedule multiplier is applied on top of these base rates:

```mermaid
graph LR
    subgraph BaseLR["Base Learning Rates"]
        E[Embedding<br/>0.3]
        U[Unembedding<br/>0.004]
        M[Matrix<br/>0.02]
        S[Scalar<br/>0.5]
    end
    
    Mult[Schedule<br/>Multiplier]
    
    E -->|×| Mult
    U -->|×| Mult
    M -->|×| Mult
    S -->|×| Mult
    
    Mult --> E_eff[Embedding<br/>effective LR]
    Mult --> U_eff[Unembedding<br/>effective LR]
    Mult --> M_eff[Matrix<br/>effective LR]
    Mult --> S_eff[Scalar<br/>effective LR]
    
    style E fill:#2d4a3e,stroke:#4aba8a,color:#e0e0e0
    style U fill:#2d4a3e,stroke:#4aba8a,color:#e0e0e0
    style M fill:#1e3a5f,stroke:#4a9eed,color:#e0e0e0
    style S fill:#5a4a2e,stroke:#d4a84b,color:#e0e0e0
    style Mult fill:#2d2d3d,stroke:#7a7a8a,color:#e0e0e0
```

<!-- Sources: scripts/base_train.py:62-66 -->

### Pretraining Base LRs

| Parameter Group | Parameters | Base LR | Optimizer | Source |
|---|---|---|---|---|
| **Embeddings** | wte | 0.3 | AdamW | [scripts/base_train.py:62](https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py#L62) |
| **Unembedding** | lm_head | 0.004 | AdamW | [scripts/base_train.py:63](https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py#L63) |
| **Matrices** | Q/K/V/proj, MLP | 0.02 | Muon | [scripts/base_train.py:65](https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py#L65) |
| **Scalars** | resid_lambdas, x0_lambdas | 0.5 | AdamW | [scripts/base_train.py:66](https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py#L66) |

The unembedding (lm_head) uses a **25× lower learning rate** than embeddings because it's a matrix multiply with high fan-out (vocab_size outputs), making it more sensitive to large updates.

### Fine-Tuning Base LRs

For SFT and RL, the base learning rates are typically scaled down and initialized at different fractions of the pretraining rates:

| Training Stage | LR Init Fraction | Typical Embedding LR | Source |
|---|---|---|---|
| **Pretraining** | 1.0 | 0.3 | Default |
| **SFT** | 1.0 (start at full) | 0.3 | [scripts/chat_sft.py:55](https://github.com/karpathy/nanochat/blob/master/scripts/chat_sft.py#L55) |
| **RL** | 0.05 (start at 5%) | 0.015 | [scripts/chat_rl.py:58](https://github.com/karpathy/nanochat/blob/master/scripts/chat_rl.py#L58) |

## Stage-Specific Schedules

```mermaid
stateDiagram-v2
    [*] --> Pretrain
    
    Pretrain: Pretraining
    Pretrain: warmup_ratio = 0.0<br/>warmdown_ratio = 0.5<br/>final_lr_frac = 0.0
    
    SFT: Supervised Fine-Tuning
    SFT: init_lr_frac = 1.0<br/>warmdown over 1 epoch<br/>weight_decay = 0.0
    
    RL: Reinforcement Learning
    RL: init_lr_frac = 0.05<br/>warmdown over 1 epoch<br/>weight_decay = 0.0
    
    Pretrain --> SFT: Load checkpoint
    SFT --> RL: Load SFT checkpoint
    RL --> [*]
    
    note right of Pretrain
        No warmup needed
        Modern optimizers are stable
        Decay to zero over second half
    end note
    
    note right of SFT
        Start at full pretrain LR
        Lower weight decay preserves
        pretrained knowledge
    end note
    
    note right of RL
        Start at 5% of pretrain LR
        Policy gradient training needs
        smaller steps for stability
    end note
```

<!-- Sources: scripts/base_train.py:68-70, scripts/chat_sft.py:50-65, scripts/chat_rl.py:44-58 -->

### Pretraining Schedule

- **Warmup**: 0% (none)
- **Constant**: First 50% of training at full LR
- **Warmdown**: Second 50% cosine decay to 0
- **Weight Decay**: 0.2 for Muon matrices, 0.0 for embeddings/scalars

### SFT Schedule

- **Init LR Frac**: 1.0 (start at full pretrained base LR)
- **Epochs**: Typically 1 epoch over SFT dataset
- **Weight Decay**: 0.0 (preserve pretrained representations)
- **Warmdown**: Cosine decay over the epoch

### RL Schedule

- **Init LR Frac**: 0.05 (start at 5% of pretrained base LR)
- **Epochs**: Typically 1 epoch over RL dataset (e.g., GSM8K train set)
- **Weight Decay**: 0.0
- **Warmdown**: Cosine decay over the epoch

The lower initial LR for RL prevents **policy collapse** — when the policy updates too aggressively and forgets the SFT initialization.

## Momentum Scheduling

Muon's **Nesterov momentum** coefficient can also be scheduled, though by default it remains constant at 0.95 throughout training:

| Phase | Muon Momentum (β₁) | AdamW Beta1 | Source |
|---|---|---|---|
| All phases | 0.95 (constant) | 0.8 (constant) | [nanochat/optim.py:261](https://github.com/karpathy/nanochat/blob/master/nanochat/optim.py#L261) |

Unlike learning rate, momentum is typically kept constant as it controls the **time scale of gradient averaging** rather than step size. However, some training regimes increase momentum during warmdown (e.g., 0.95 → 0.99) to smooth out the final convergence.

## Schedule Visualization

```mermaid
graph TB
    subgraph Timeline["Training Timeline (10,000 iterations)"]
        T0[Iteration 0<br/>LR = 0.0]
        T1[Iteration 0<br/>LR = 1.0 * base<br/>Constant phase starts]
        T2[Iteration 5,000<br/>LR = 1.0 * base<br/>Warmdown starts]
        T3[Iteration 7,500<br/>LR = 0.5 * base<br/>Cosine midpoint]
        T4[Iteration 10,000<br/>LR = 0.0 * base<br/>Training ends]
    end
    
    T0 --> T1
    T1 --> T2
    T2 --> T3
    T3 --> T4
    
    T1 -.->|No warmup<br/>warmup_ratio=0.0| T1
    T2 -.->|50% of training<br/>warmdown_ratio=0.5| T4
    T4 -.->|Decay to zero<br/>final_lr_frac=0.0| T4
    
    style T0 fill:#4a2e2e,stroke:#d45b5b,color:#e0e0e0
    style T1 fill:#2d4a3e,stroke:#4aba8a,color:#e0e0e0
    style T2 fill:#1e3a5f,stroke:#4a9eed,color:#e0e0e0
    style T3 fill:#5a4a2e,stroke:#d4a84b,color:#e0e0e0
    style T4 fill:#4a2e2e,stroke:#d45b5b,color:#e0e0e0
```

<!-- Sources: scripts/base_train.py:68-70 -->

## Implementation Notes

### Avoiding Recompilation

The schedule multiplier is computed on the **host (CPU)** and passed to the optimizer as a 0-D CPU tensor. This avoids `torch.compile` recompilation when the LR changes, since the compiled kernels treat the LR as a runtime input rather than a constant ([nanochat/optim.py:182-192](https://github.com/karpathy/nanochat/blob/master/nanochat/optim.py#L182-L192)).

### Distributed Synchronization

In distributed training, each rank computes the same schedule independently — no cross-rank communication is needed for LR scheduling. This avoids collective communication overhead and ensures all ranks stay synchronized by construction.

### Checkpoint Resumption

When resuming from a checkpoint, the schedule continues from the saved iteration count. The `--resume-from-step` flag allows manual override if needed ([scripts/base_train.py:71](https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py#L71)).

## Why This Schedule?

```mermaid
graph TB
    subgraph Design["Schedule Design Rationale"]
        N1[No Warmup Needed]
        N2[Modern optimizers stable<br/>LayerNorm prevents explosions]
        
        C1[Long Constant Phase]
        C2[Majority of progress<br/>happens at full LR]
        
        D1[Cosine Warmdown]
        D2[Smooth convergence<br/>to sharper minima]
        D3[Decay to zero<br/>exhausts budget]
    end
    
    N1 --> N2
    C1 --> C2
    D1 --> D2
    D2 --> D3
    
    N2 --> Result[Efficient<br/>Training]
    C2 --> Result
    D3 --> Result
    
    style N1 fill:#2d4a3e,stroke:#4aba8a,color:#e0e0e0
    style C1 fill:#1e3a5f,stroke:#4a9eed,color:#e0e0e0
    style D1 fill:#5a4a2e,stroke:#d4a84b,color:#e0e0e0
    style Result fill:#2d4a3e,stroke:#4aba8a,color:#e0e0e0
```

<!-- Sources: scripts/base_train.py:68-70, README.md:78 -->

The schedule design reflects several insights:
1. **No warmup**: Modern architectures with LayerNorm are stable from iteration 0
2. **50% warmdown**: Allows ample time at full LR for the bulk of optimization
3. **Decay to zero**: Ensures the compute budget is fully exhausted (no "leftover" gradient steps with nearly-zero LR)
4. **Per-group LRs**: Different parameter types have different optimal step sizes
5. **Cosine smoothness**: Avoids abrupt LR changes that can destabilize training

## References

- [Cosine Annealing paper (arxiv:1608.03983)](https://arxiv.org/abs/1608.03983) — Ilya Loshchilov, Frank Hutter
- [Adam paper (arxiv:1412.6980)](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma, Jimmy Ba
- [nanochat pretraining script](https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py) — Default schedule parameters
- [nanochat SFT script](https://github.com/karpathy/nanochat/blob/master/scripts/chat_sft.py) — Fine-tuning LR configuration
- [nanochat RL script](https://github.com/karpathy/nanochat/blob/master/scripts/chat_rl.py) — Policy gradient LR configuration
