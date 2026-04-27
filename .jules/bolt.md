## 2024-05-18 - Avoid overhead with set_to_none=True in zero_grad
**Learning:** PyTorch by default instantiates a zero tensor for every parameter during `optimizer.zero_grad()`. This creates overhead.
**Action:** Always use `optimizer.zero_grad(set_to_none=True)` during backwards pass.
