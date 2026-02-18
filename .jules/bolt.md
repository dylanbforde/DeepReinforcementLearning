## 2024-02-18 - [PyTorch Soft Update Optimization]
**Learning:** PyTorch's `target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)` is inefficient because it creates multiple temporary tensors for the arithmetic operations before copying. In-place operations (`mul_`, `add_`) avoid this allocation overhead.
**Action:** When performing parameter updates (like soft updates in RL), always prefer in-place operations to minimize memory allocation and GPU/CPU sync overhead.
