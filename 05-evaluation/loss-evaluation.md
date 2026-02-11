# Loss Evaluation (BPB)

Bits per byte (BPB) is a **tokenization-invariant** metric for evaluating language models. Unlike standard cross-entropy loss (which depends on vocabulary size and tokenization granularity), BPB normalizes by the number of raw bytes in the target text, making it possible to fairly compare models with different tokenizers.

> **Source:** [`nanochat/loss_eval.py`](../../nanochat/loss_eval.py)

---

## Formula

```
BPB = total_nats / (ln(2) × total_bytes)
```

Where:
- **total_nats** — sum of per-token cross-entropy losses (in nats, i.e., natural log units)
- **total_bytes** — sum of byte lengths of all target tokens
- **ln(2)** — converts nats to bits

If `total_bytes == 0`, the function returns `float('inf')`.

---

## The `token_bytes` Tensor

`evaluate_bpb` requires a pre-computed `token_bytes` tensor of shape `(vocab_size,)`:

| Token Type | `token_bytes` Value | Effect |
|---|---|---|
| Normal token (e.g., `"hello"`) | `5` (byte length) | Counted in both nats and bytes |
| Special token (e.g., `<\|bos\|>`) | `0` | Excluded from both nats and bytes |

This ensures special tokens don't distort the metric. The loss for a token with `token_bytes == 0` is multiplied by zero, effectively masking it out.

---

## Handling Masked Tokens

Some training setups use `ignore_index=-1` to mask certain target positions (e.g., prompt tokens during SFT). The function handles this with two code paths:

### Fast Path (no masked tokens)

```python
num_bytes2d = token_bytes[y]
total_nats += (loss2d * (num_bytes2d > 0)).sum()
total_bytes += num_bytes2d.sum()
```

### Safe Path (masked tokens present)

When any target token ID is negative:

```python
valid = y >= 0
y_safe = torch.where(valid, y, torch.zeros_like(y))
num_bytes2d = torch.where(valid, token_bytes[y_safe], torch.zeros_like(...))
```

This avoids indexing `token_bytes` with negative values (which would cause out-of-bounds errors). The validity check uses `y.int() < 0` because MPS does not support the `< 0` kernel for `int64`.

---

## Evaluation Loop

```mermaid
flowchart TD
    BATCH["Batch (x, y)"] --> MODEL["model(x, y, reduction=none)<br>Per-token loss"]
    MODEL --> BYTES["token_bytes[y]<br>Bytes per target token"]
    BYTES --> VALID{"token valid?<br>bytes > 0, y >= 0"}
    VALID -->|Yes| ACC["total_nats += loss<br>total_bytes += bytes"]
    VALID -->|No| SKIP["Skip<br>(special tokens, padding)"]
    ACC --> REDUCE["all_reduce across ranks"]
    REDUCE --> BPB["bpb = total_nats<br>/ (ln(2) * total_bytes)"]

    style BATCH fill:#2d333b,stroke:#6d5dfc,color:#e6edf3
    style MODEL fill:#2d333b,stroke:#6d5dfc,color:#e6edf3
    style BYTES fill:#2d333b,stroke:#6d5dfc,color:#e6edf3
    style VALID fill:#2d333b,stroke:#6d5dfc,color:#e6edf3
    style ACC fill:#2d333b,stroke:#6d5dfc,color:#e6edf3
    style SKIP fill:#2d333b,stroke:#8b949e,color:#e6edf3
    style REDUCE fill:#2d333b,stroke:#6d5dfc,color:#e6edf3
    style BPB fill:#2d333b,stroke:#6d5dfc,color:#e6edf3
```

The function iterates over a fixed number of batches:

```python
for _ in range(steps):
    x, y = next(batch_iter)
    loss2d = model(x, y, loss_reduction='none')  # (B, T) per-token losses
    # accumulate total_nats and total_bytes
```

Key details:
- The model is called with `loss_reduction='none'` to get per-token losses instead of a scalar mean
- Both `loss2d` and `y` are flattened before processing

---

## Distributed Evaluation

When running with `torchrun`, partial sums are aggregated across all ranks:

```python
if world_size > 1:
    dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
```

Each rank processes its own batches independently, then `total_nats` and `total_bytes` are summed before computing the final BPB. This ensures the metric is identical regardless of the number of GPUs.

---

## Why BPB Over Cross-Entropy?

```mermaid
flowchart LR
    subgraph NaiveLoss["Naive Loss"]
        NL["mean(cross_entropy)<br>Varies with vocab size"]
    end
    subgraph BPBMetric["Bits Per Byte"]
        BP["sum(loss) / sum(bytes)<br>Vocab-independent"]
    end
    NaiveLoss -->|"32K vocab ≠ 50K vocab"| PROBLEM["Not comparable"]
    BPBMetric -->|"Normalized by bytes"| FAIR["Apples-to-apples"]

    style NaiveLoss fill:#161b22,stroke:#30363d,color:#e6edf3
    style BPBMetric fill:#161b22,stroke:#30363d,color:#e6edf3
    style NL fill:#2d333b,stroke:#8b949e,color:#e6edf3
    style BP fill:#2d333b,stroke:#6d5dfc,color:#e6edf3
    style PROBLEM fill:#2d333b,stroke:#8b949e,color:#e6edf3
    style FAIR fill:#2d333b,stroke:#6d5dfc,color:#e6edf3
```

| Metric | Depends on Vocab Size | Depends on Tokenizer | Comparable Across Models |
|---|---|---|---|
| Cross-entropy (nats/token) | ✅ Yes | ✅ Yes | ❌ No |
| Perplexity | ✅ Yes | ✅ Yes | ❌ No |
| **Bits per byte** | ❌ No | ❌ No | ✅ Yes |

A model with a larger vocabulary has shorter sequences (fewer tokens per text), so its per-token loss appears lower even if it isn't actually better. BPB eliminates this confound by normalizing to the fixed, universal unit of bytes.
