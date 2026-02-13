---
title: Inference Engine
description: Efficient KV cache engine with tool use (calculator) support and batch generation
outline: deep
---

# Inference Engine

The `Engine` class provides efficient batch generation with KV cache reuse, tool use via calculator, and streaming output.

## Components

| Component | Purpose | Source |
|---|---|---|
| **KVCache** | Past key/value tensors in FA3 BHLD layout | [nanochat/engine.py:83-133](https://github.com/karpathy/nanochat/blob/master/nanochat/engine.py#L83-L133) |
| **RowState** | Per-row state (tokens, forced tokens, tool use) | [nanochat/engine.py:155-163](https://github.com/karpathy/nanochat/blob/master/nanochat/engine.py#L155-L163) |
| **Calculator** | Safe eval with timeout | [nanochat/engine.py:26-81](https://github.com/karpathy/nanochat/blob/master/nanochat/engine.py#L26-L81) |

```mermaid
graph TB
    subgraph Generation
        P[Prefill prompt<br/>batch=1]
        R[Replicate KV cache]
        S[Sample next token]
        T[Tool use logic]
    end
    
    P --> R
    R --> S
    S --> T
    T --> S
    
    style P fill:#1e3a5f,stroke:#4a9eed,color:#e0e0e0
    style S fill:#2d4a3e,stroke:#4aba8a,color:#e0e0e0
    style T fill:#5a4a2e,stroke:#d4a84b,color:#e0e0e0
```

<!-- Sources: nanochat/engine.py:171-276 -->

## Tool State Machine

```mermaid
stateDiagram-v2
    [*] --> Normal
    Normal --> PythonBlock: <|python_start|>
    PythonBlock --> Calculate: <|python_end|>
    Calculate --> Normal: Inject result
```

<!-- Sources: nanochat/engine.py:252-267 -->

Calculator supports math and `.count()` with 3-second timeout ([nanochat/engine.py:47-80](https://github.com/karpathy/nanochat/blob/master/nanochat/engine.py#L47-L80)).

## References

- [nanochat/engine.py](https://github.com/karpathy/nanochat/blob/master/nanochat/engine.py)
