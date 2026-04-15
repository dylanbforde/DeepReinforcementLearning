## 2024-04-15 - [PyTorch Soft Target Updates Bottleneck]
**Learning:** Manual interpolation for soft target network updates (`target = tau * policy + (1-tau) * target`) creates multiple intermediate tensors. In PyTorch training loops, this introduces unnecessary memory allocations and CPU-GPU syncs during every optimization step, significantly slowing down training.
**Action:** Always use PyTorch's in-place operations like `.lerp_()` (`target.lerp_(policy, tau)`) for soft updates. It achieves mathematical equivalence without allocating intermediate tensors, drastically improving performance.
