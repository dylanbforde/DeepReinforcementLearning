
## 2024-03-15 - [PyTorch Tensor Operations Performance: lerp_ vs Math]
**Learning:** Performing math operations like `tau * policy_param.data + (1.0 - tau) * target_param.data` in PyTorch, even inside `.copy_()`, creates multiple intermediate tensor allocations which are slow.
**Action:** Use native PyTorch in-place tensor operations whenever possible. For soft-updating target network weights, `target_param.data.lerp_(policy_param.data, tau)` avoids intermediate allocations and is approximately 3x faster than manually calculating the math expression.
