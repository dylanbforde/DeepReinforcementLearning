## 2024-05-18 - [PyTorch Tensor Operations]
**Learning:** Out-of-place soft updates for target networks using `.copy_` and basic arithmetic create intermediate tensors, unnecessarily consuming memory and increasing CPU-GPU synchronization overhead during the optimization loop.
**Action:** Always use in-place `.lerp_` for soft updating target network parameters to optimize memory allocation and synchronization times.
