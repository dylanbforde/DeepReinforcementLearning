## 2024-11-20 - [Soft Update Memory Optimization]
**Learning:** The memory optimization with `lerp_` has already been implemented in `OptimizeModel.py` and is in memory as well, so let's look for another bottleneck.
**Action:** Search for other pre-allocation or missing caching opportunities.
## 2024-11-20 - [Action Selection Generation Optimization]
**Learning:** `np.random.uniform` wrapped in `torch.tensor` causes CPU-to-GPU overhead (or extra allocations) during action selection, but I remember this is already optimized to `torch.empty(...).uniform_(-1, 1)` in the training loop of `Main.py`. But wait, `ActionSelection.py` still uses it!
**Action:** Let's check `Main.py` to see if the action selection optimization was applied everywhere.
## 2024-11-20 - [Zero Grad vs Set to None]
**Learning:** `set_to_none=True` doesn't necessarily improve speed over `zero_grad()` on this architecture, possibly due to memory reallocation overhead.
**Action:** Ignore `set_to_none=True` for speed improvement here.
