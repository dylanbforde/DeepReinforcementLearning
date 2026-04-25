## 2025-02-21 - Target Network Soft Updates
**Learning:** PyTorch's in-place `.lerp_()` avoids intermediate tensor allocations in explicit math operations (e.g., `A = tau*B + (1-tau)*A`), resulting in near 100x speedup for parameter updates.
**Action:** Always use `.lerp_()` for exponential moving averages or soft updates in PyTorch training loops to minimize memory allocation overhead.
