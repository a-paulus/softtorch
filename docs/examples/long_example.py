"""Quick Example — prints all outputs for README and documentation."""

import softtorch as st
import torch


torch.set_printoptions(precision=4, sci_mode=False)

# ── Elementwise operators ─────────────────────────────────────────────────────

x = torch.tensor([-0.2, -1.0, 0.3, 1.0])

print("\nTorch absolute:", torch.abs(x))
print("SoftTorch absolute (hard mode):", st.abs(x, mode="hard"))
print("SoftTorch absolute (soft mode):", st.abs(x))

print("\nTorch clamp:", torch.clamp(x, -0.5, 0.5))
print("SoftTorch clamp (hard mode):", st.clamp(x, -0.5, 0.5, mode="hard"))
print("SoftTorch clamp (soft mode):", st.clamp(x, -0.5, 0.5))

print("\nTorch heaviside:", torch.heaviside(x, torch.tensor(0.5)))
print("SoftTorch heaviside (hard mode):", st.heaviside(x, mode="hard"))
print("SoftTorch heaviside (soft mode):", st.heaviside(x))

print("\nTorch ReLU:", torch.nn.functional.relu(x))
print("SoftTorch ReLU (hard mode):", st.relu(x, mode="hard"))
print("SoftTorch ReLU (soft mode):", st.relu(x))

print("\nTorch round:", torch.round(x))
print("SoftTorch round (hard mode):", st.round(x, mode="hard"))
print("SoftTorch round (soft mode):", st.round(x))

print("\nTorch sign:", torch.sign(x))
print("SoftTorch sign (hard mode):", st.sign(x, mode="hard"))
print("SoftTorch sign (soft mode):", st.sign(x))

# ── Tensor-valued operators ──────────────────────────────────────────────────

print("\nTorch max:", torch.max(x))
print("SoftTorch max (hard mode):", st.max(x, mode="hard"))
print("SoftTorch max (soft mode):", st.max(x))

print("\nTorch min:", torch.min(x))
print("SoftTorch min (hard mode):", st.min(x, mode="hard"))
print("SoftTorch min (soft mode):", st.min(x))

print("\nTorch sort:", torch.sort(x).values)
print("SoftTorch sort (hard mode):", st.sort(x, mode="hard").values)
print("SoftTorch sort (soft mode):", st.sort(x).values)

print("\nTorch quantile:", torch.quantile(x, q=0.2))
print("SoftTorch quantile (hard mode):", st.quantile(x, q=0.2, mode="hard"))
print("SoftTorch quantile (soft mode):", st.quantile(x, q=0.2))

print("\nTorch median:", torch.median(x))
print("SoftTorch median (hard mode):", st.median(x, mode="hard"))
print("SoftTorch median (soft mode):", st.median(x))

print("\nTorch topk:", torch.topk(x, k=3).values)
print("SoftTorch topk (hard mode):", st.topk(x, k=3, mode="hard").values)
print("SoftTorch topk (soft mode):", st.topk(x, k=3).values)

print("\nTorch rank:", torch.argsort(torch.argsort(x)))
print("SoftTorch rank (hard mode):", st.rank(x, mode="hard", descending=False))
print("SoftTorch rank (soft mode):", st.rank(x, descending=False))

# ── Sort: sweep over methods ──────────────────────────────────────────────────

print("\nTorch sort:", torch.sort(x).values)
print("SoftTorch sort (softsort):", st.sort(x, method="softsort", softness=0.1).values)
print("SoftTorch sort (neuralsort):", st.sort(x, method="neuralsort", softness=0.1).values)
print("SoftTorch sort (fast_soft_sort):", st.sort(x, method="fast_soft_sort", softness=2.0).values)
print("SoftTorch sort (ot):", st.sort(x, method="ot", softness=0.1).values)
print("SoftTorch sort (sorting_network):", st.sort(x, method="sorting_network", softness=0.1).values)

# ── Sort: sweep over modes ──────────────────────────────────────────────────

print("\nTorch sort:", torch.sort(x).values)
for mode in ["hard", "smooth", "c0", "c1", "c2"]:
    print(f"SoftTorch sort ({mode}):", st.sort(x, softness=0.5, mode=mode).values)

# ── Operators returning indices ──────────────────────────────────────────────

print("\nTorch argmax:", torch.argmax(x))
print("SoftTorch argmax (hard mode):", st.argmax(x, mode="hard"))
print("SoftTorch argmax (soft mode):", st.argmax(x))

print("\nTorch argmin:", torch.argmin(x))
print("SoftTorch argmin (hard mode):", st.argmin(x, mode="hard"))
print("SoftTorch argmin (soft mode):", st.argmin(x))

print("\nTorch argquantile:", "Not implemented in standard PyTorch")
print("SoftTorch argquantile (hard mode):", st.argquantile(x, q=0.2, mode="hard"))
print("SoftTorch argquantile (soft mode):", st.argquantile(x, q=0.2))

print("\nTorch argmedian:", torch.median(x, dim=0).indices)
print("SoftTorch argmedian (hard mode):", st.median(x, mode="hard", dim=0).indices)
print("SoftTorch argmedian (soft mode):", st.median(x, dim=0).indices)

print("\nTorch argsort:", torch.argsort(x))
print("SoftTorch argsort (hard mode):", st.argsort(x, mode="hard"))
print("SoftTorch argsort (soft mode):", st.argsort(x))

print("\nTorch argtopk:", torch.topk(x, k=3).indices)
print("SoftTorch argtopk (hard mode):", st.topk(x, k=3, mode="hard").indices)
print("SoftTorch argtopk (soft mode):", st.topk(x, k=3).indices)

# ── Comparison operators ────────────────────────────────────────────────────

y = torch.tensor([0.2, -0.5, 0.5, -1.0])

print("\nTorch greater:", torch.greater(x, y))
print("SoftTorch greater (hard mode):", st.greater(x, y, mode="hard"))
print("SoftTorch greater (soft mode):", st.greater(x, y))

print("\nTorch greater equal:", torch.greater_equal(x, y))
print("SoftTorch greater equal (hard mode):", st.greater_equal(x, y, mode="hard"))
print("SoftTorch greater equal (soft mode):", st.greater_equal(x, y))

print("\nTorch less:", torch.less(x, y))
print("SoftTorch less (hard mode):", st.less(x, y, mode="hard"))
print("SoftTorch less (soft mode):", st.less(x, y))

print("\nTorch less equal:", torch.less_equal(x, y))
print("SoftTorch less equal (hard mode):", st.less_equal(x, y, mode="hard"))
print("SoftTorch less equal (soft mode):", st.less_equal(x, y))

print("\nTorch eq:", torch.eq(x, y))
print("SoftTorch eq (hard mode):", st.eq(x, y, mode="hard"))
print("SoftTorch eq (soft mode):", st.eq(x, y))

print("\nTorch not equal:", torch.not_equal(x, y))
print("SoftTorch not equal (hard mode):", st.not_equal(x, y, mode="hard"))
print("SoftTorch not equal (soft mode):", st.not_equal(x, y))

print("\nTorch isclose:", torch.isclose(x, y))
print("SoftTorch isclose (hard mode):", st.isclose(x, y, mode="hard"))
print("SoftTorch isclose (soft mode):", st.isclose(x, y))

# ── Logical operators ───────────────────────────────────────────────────────

fuzzy_a = torch.tensor([0.1, 0.2, 0.8, 1.0])
fuzzy_b = torch.tensor([0.7, 0.3, 0.1, 0.9])
bool_a = fuzzy_a >= 0.5
bool_b = fuzzy_b >= 0.5

print("\nTorch AND:", torch.logical_and(bool_a, bool_b))
print("SoftTorch AND:", st.logical_and(fuzzy_a, fuzzy_b))

print("\nTorch OR:", torch.logical_or(bool_a, bool_b))
print("SoftTorch OR:", st.logical_or(fuzzy_a, fuzzy_b))

print("\nTorch NOT:", torch.logical_not(bool_a))
print("SoftTorch NOT:", st.logical_not(fuzzy_a))

print("\nTorch XOR:", torch.logical_xor(bool_a, bool_b))
print("SoftTorch XOR:", st.logical_xor(fuzzy_a, fuzzy_b))

print("\nTorch ALL:", torch.all(bool_a))
print("SoftTorch ALL:", st.all(fuzzy_a))

print("\nTorch ANY:", torch.any(bool_a))
print("SoftTorch ANY:", st.any(fuzzy_a))

# Selection operators
print("\nTorch Where:", torch.where(bool_a, x, y))
print("SoftTorch Where:", st.where(fuzzy_a, x, y))

# ── Straight-through operators ──────────────────────────────────────────────

print("\nStraight-through ReLU:", st.relu_st(x))
print("Straight-through sort:", st.sort_st(x).values)
print("Straight-through argtopk:", st.topk_st(x, k=3).indices)
print("Straight-through greater:", st.greater_st(x, y))
