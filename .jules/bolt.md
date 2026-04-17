## 2024-05-15 - [Direct PyTorch Tensor Allocation]
**Learning:** In PyTorch workloads dealing with continuous action spaces, creating arrays with `np.random.uniform` and casting them to tensors generates unnecessary CPU allocations and requires transfer to the target device.
**Action:** Use `torch.empty(..., device=device).uniform_(...)` to directly sample continuous distributions in-place on the target device natively, bypassing CPU numpy intermediaries.
