## 2026-03-18 - PyTorch Soft Update Optimization
**Learning:** Manual tensor arithmetic for PyTorch target network updates (`target.data.copy_(tau * policy.data + (1-tau) * target.data)`) creates multiple intermediate tensors and allocations per parameter per step, which is highly inefficient.
**Action:** Use PyTorch's in-place linear interpolation `lerp_` method (`target.data.lerp_(policy.data, tau)`) for soft network updates to perform the operation in-place and avoid intermediate allocations, resulting in significant (~10x for this operation) performance gains.
