## 2024-05-18 - [Optimize PyTorch target network soft updates]
**Learning:** Performing a manual soft update (e.g. `tau * policy + (1-tau) * target`) creates intermediate tensors which slows down PyTorch's parameter update process in optimization loops.
**Action:** Use `.lerp_()` on PyTorch tensor data for in-place linear interpolation, accelerating the backward/update pass by avoiding intermediate allocations.
