---
title: Sandboxed Code Execution
description: Process-isolated Python execution with timeout and memory limits
outline: deep
---

# Sandboxed Code Execution

Safe execution for model-generated code with process isolation, timeouts, and memory limits.

## Safety Layers

```mermaid
graph TB
    Code[Generated Code]
    P[Process isolation]
    T[Timeout 5s]
    M[Memory limit 256MB]
    R[Reliability guard]
    
    Code --> P
    P --> T
    P --> M
    P --> R
    
    style P fill:#1e3a5f,stroke:#4a9eed,color:#e0e0e0
    style R fill:#4a2e2e,stroke:#d45b5b,color:#e0e0e0
```

<!-- Sources: nanochat/execution.py:1-22 -->

## Blocked Functions

| Category | Functions | Source |
|---|---|---|
| File system | `os.remove`, `shutil.rmtree` | [nanochat/execution.py:164-191](https://github.com/karpathy/nanochat/blob/master/nanochat/execution.py#L164-L191) |
| Process | `os.system`, `subprocess.Popen` | [nanochat/execution.py:162-201](https://github.com/karpathy/nanochat/blob/master/nanochat/execution.py#L162-L201) |

## Usage

```python
from nanochat.execution import execute_code

result = execute_code("print('hello')", timeout=5.0)
print(result.success, result.stdout)
```

## Limitations

**Not a true security sandbox** — protects against accidents, not adversarial code ([nanochat/execution.py:14-22](https://github.com/karpathy/nanochat/blob/master/nanochat/execution.py#L14-L22)).

## References

- [nanochat/execution.py](https://github.com/karpathy/nanochat/blob/master/nanochat/execution.py)
