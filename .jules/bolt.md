## 2025-02-14 - PyTorch Tensor Allocation Anti-Pattern
**Learning:** Using explicit arithmetic `target.copy_(tau * policy + (1 - tau) * target)` in soft updates creates intermediate tensors, leading to memory overhead. `target.lerp_(policy, tau)` is a direct in-place alternative. Additionally, `zero_grad()` without `set_to_none=True` leaves tensors in memory.
**Action:** Always use `.lerp_()` for soft updates and `zero_grad(set_to_none=True)` in PyTorch training loops to avoid unnecessary allocations.
