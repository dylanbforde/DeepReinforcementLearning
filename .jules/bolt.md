
## 2024-04-11 - PyTorch In-place Operations
**Learning:** Target network soft updates in DQN (`target = tau * policy + (1-tau) * target`) are a significant bottleneck when using explicit arithmetic due to intermediate tensor allocations on the CPU/GPU, even with `.data` access.
**Action:** Always utilize PyTorch's in-place operations like `.lerp_()` (Linear Interpolation) when computing convex combinations or updating weights. It eliminates intermediate allocations and runs significantly faster (~96% speedup).
