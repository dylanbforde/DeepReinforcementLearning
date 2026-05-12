## 2024-05-12 - PyTorch Soft Update Optimization
**Learning:** PyTorch's in-place `.lerp_()` is vastly more efficient for soft updates on this architecture due to the avoidance of intermediate tensor allocations compared to `.copy_()` with explicit arithmetic.
**Action:** Always prefer `.lerp_()` for soft updating target networks in PyTorch to reduce memory reallocation overhead.
