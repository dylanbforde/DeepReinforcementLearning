## 2024-10-24 - PyTorch in-place tensor operations for soft updates
**Learning:** Using explicit arithmetic `target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)` creates intermediate tensors, leading to high allocation overhead and slower runtime on hot paths (like training loops).
**Action:** Use `.lerp_()` which is implemented by PyTorch natively in C++ for in-place linear interpolation, avoiding any intermediate memory allocations. `target_param.data.lerp_(policy_param.data, TAU)`
