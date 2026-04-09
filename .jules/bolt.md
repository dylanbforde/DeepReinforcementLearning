## $(date +%Y-%m-%d) - Optimize Target Network Soft Updates
**Learning:** In PyTorch, using explicit arithmetic expressions (e.g. `A = tau * B + (1 - tau) * A`) for parameter updates allocates multiple intermediate tensors per parameter, causing significant memory and performance overhead in tight training loops.
**Action:** Always prefer PyTorch's built-in, in-place update functions like `.lerp_()` when performing linear interpolations, especially for target network soft updates in RL.
