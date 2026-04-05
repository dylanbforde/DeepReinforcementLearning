## 2025-02-23 - [PyTorch Tensor Operations]
**Learning:** Target network soft updates utilizing PyTorch's in-place `.lerp_()` avoid intermediate tensor allocations, yielding significant performance improvements over `.copy_()` with explicit arithmetic.
**Action:** Always prefer PyTorch's optimized, in-place mathematical functions (like `.lerp_()`, `.add_()`, `.mul_()`) over combining basic mathematical operators (`+`, `*`) that inherently spawn short-lived, intermediate tensors.
