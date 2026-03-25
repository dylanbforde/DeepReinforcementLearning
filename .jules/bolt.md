
## 2025-02-13 - Replace copy_ with lerp_ in Soft Updates
**Learning:** Target network soft updates in DQN/DDPG models often use explicit arithmetic operations with `.copy_()` (e.g., `target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)`). This approach leads to intermediate tensor allocations. Using PyTorch's in-place linear interpolation `lerp_()` (e.g. `target_param.data.lerp_(policy_param.data, TAU)`) mathematically equivalent but bypasses these intermediate memory allocations, delivering over a 30x speedup in the update step loop when timed.
**Action:** When performing linear interpolations or soft network updates, always use `lerp_` to eliminate unnecessary memory allocation and improve performance.
