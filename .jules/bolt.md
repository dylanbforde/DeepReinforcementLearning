## 2024-05-18 - Target Network Soft Updates: lerp_() vs copy_()
**Learning:** Using `target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)` creates intermediate tensors for `TAU * policy_param.data` and `(1.0 - TAU) * target_param.data` before copying. This causes unnecessary memory allocation and computation overhead.
**Action:** Always use PyTorch's in-place `lerp_()` (`target_param.data.lerp_(policy_param.data, TAU)`) for soft updates to perform the interpolation in-place without intermediate tensor allocations, significantly improving performance.
