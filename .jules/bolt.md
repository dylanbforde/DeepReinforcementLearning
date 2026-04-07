## 2024-03-27 - PyTorch Soft Update Optimization
**Learning:** Explicit arithmetic in target network soft updates (`target.copy_(tau * policy + (1-tau) * target)`) creates multiple intermediate tensors per parameter every step, leading to significant memory allocation overhead.
**Action:** Use PyTorch's in-place `.lerp_()` method (`target.lerp_(policy, tau)`) for soft updates to avoid intermediate allocations and speed up the training loop.
