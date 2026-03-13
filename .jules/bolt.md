## 2024-05-18 - [PyTorch Performance Anti-Patterns]
**Learning:** Manual soft update calculation (`target.data.copy_(tau * policy.data + (1-tau) * target.data)`) is an anti-pattern that creates intermediate tensors and is ~15x slower than PyTorch's in-place `lerp_()`.
**Action:** Always use `tensor.lerp_()` for target network soft updates and `optimizer.zero_grad(set_to_none=True)` to avoid unnecessary memory allocation during training loops.
