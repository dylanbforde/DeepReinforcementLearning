
## $(date +%Y-%m-%d) - PyTorch In-Place Gradients
**Learning:** `optimizer.zero_grad(set_to_none=True)` in PyTorch is significantly faster than standard zero_grad because it doesn't execute memory operations to set gradients to zero but rather deletes the gradient tensors.
**Action:** Always prefer `set_to_none=True` when zeroing out gradients unless there's a specific requirement for zeroed gradient buffers.

## $(date +%Y-%m-%d) - Avoiding Redundant Forward Passes
**Learning:** Pre-computed tensors (like `predicted_suitability`) should be passed into training/update functions when calculating loss, rather than forcing the model to run a second forward pass.
**Action:** Audit training loops for redundant forward passes where the output was already calculated for inference/action-selection.
