## 2024-05-24 - PyTorch Random Array Generation on Device
**Learning:** Initializing random data through NumPy (e.g., `np.random.uniform`) before converting it to PyTorch tensors in tight loops is an architectural bottleneck. It forces allocation on the CPU followed by data transfer to the GPU, causing severe pipeline stalls (approx 2.4x slowdown on continuous action sampling).
**Action:** Always generate random continuous tensors directly on the target device using `torch.empty(...).uniform_(lower, upper)`.
