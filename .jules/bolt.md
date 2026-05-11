## 2024-05-18 - PyTorch Target Network Update Speedup
**Learning:** Manual calculation in target network soft updates (`target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)`) creates intermediate tensor allocations and is significantly slower than PyTorch's built-in `lerp_` which does it in-place.
**Action:** Use `t.data.lerp_(p.data, tau)` for all soft network parameter updates.
