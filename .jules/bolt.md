## 2024-05-18 - [DataLogger CSV Writer Optimization]
**Learning:** Using csv.writer with tuples provides a ~1.5x speedup compared to csv.DictWriter with dictionaries due to avoided key-lookup overhead during large-scale iterative logging operations in reinforcement learning.
**Action:** Prefer csv.writer with tuples over csv.DictWriter with dictionaries when logging large amounts of data iteratively to a CSV file.
