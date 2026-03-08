
## 2024-03-24 - PyTorch In-Place Soft Update Optimization
**Learning:** Manual arithmetic for soft updating target networks (`TAU * policy_param + (1 - TAU) * target_param`) creates multiple intermediate tensors. `target_param.data.lerp_(policy_param.data, TAU)` provides the exact same mathematical result in-place without the intermediate tensor allocations, which can lead to a 10-20x speedup in the soft update loop. Also `set_to_none=True` on `zero_grad` prevents allocating tensors filled with zeros.
**Action:** Always prefer `lerp_` for exponential moving averages and `set_to_none=True` on `zero_grad` in PyTorch training loops to minimize memory allocations. Ensure not to commit `.pyc` files that are generated when benchmarking/running tests.
