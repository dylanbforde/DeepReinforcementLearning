## 2025-02-21 - Memory allocations in target network soft updates
**Learning:** Target network soft updates are typically implemented as `target.copy_(tau * policy + (1 - tau) * target)`. In PyTorch, this triggers unnecessary intermediate tensor allocations behind the scenes.
**Action:** Always prefer the in-place equivalent `target.lerp_(policy, tau)` which computes exactly the same values without creating intermediate objects, thereby reducing memory churn and execution time overhead in the training loop.
