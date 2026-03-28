## 2024-05-24 - [PyTorch Tensor Copy Overhead]
**Learning:** Manual arithmetic mixed with `.copy_()` (e.g., `target.copy_(tau * policy + (1-tau) * target)`) causes severe performance bottlenecks due to intermediate tensor allocations. This is highly visible inside inner training loops like target network soft updates in Reinforcement Learning algorithms.
**Action:** Use completely in-place linear interpolation methods provided by PyTorch, such as `.lerp_()`, which completely avoids allocating intermediate tensors and results in significantly faster tensor operations.
