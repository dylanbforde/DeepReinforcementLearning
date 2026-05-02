## 2024-05-15 - Synchronous CPU-GPU Data Transfer Bottleneck
**Learning:** PyTorch operations like `.item()` inside hot training loops (e.g. `reward.item()` or `loss.item()`) force a synchronous CPU-GPU data transfer on every iteration, severely degrading performance.
**Action:** Handle values like `reward` natively as standard Python floats inside the training loop, only wrapping them into tensors right before they are needed for models or memory buffers.

## 2024-05-15 - Redundant Tensor Allocations in RL Loops
**Learning:** Constant tensors like true/false labels for suitability, or duplicate tensor copies created for `state` and `next_state` in consecutive environment steps, waste significant time on allocations and device transfers.
**Action:** Pre-allocate constant tensors outside the episode loop, and use `torch.as_tensor()` combined with re-assigning variables (e.g. `state = next_state`) to recycle memory buffers.
