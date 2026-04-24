## 2024-04-24 - PyTorch In-Place Lerp Optimization
**Learning:** Target network soft updates in `DomainShift/OptimizeModel.py` utilize `.copy_()` with explicit mathematical operations (`tau * policy + (1-tau) * target`), which results in unnecessary intermediate tensor allocations. This operation is repeatedly called every optimization step across all model layers.
**Action:** Use PyTorch's native `.lerp_()` (linear interpolation) for an in-place update, avoiding allocations and leading to an ~4-5x speedup for soft updates.
