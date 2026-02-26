"""Paper examples — prints outputs used in the paper figures and text."""

import softtorch as st
import torch


torch.set_printoptions(precision=4, sci_mode=False)

# ── Soft comparison + where ─────────────────────────────────────────────────

x = torch.tensor([0.1, 0.3, 0.7])
y = torch.tensor([0.4, 0.2, 0.3])

cond = torch.greater(x, y)
print("cond:", cond)
z = torch.where(cond, x, y)
print("z:", z)

soft_cond = st.greater(x, y)
print("soft_cond:", soft_cond.round(decimals=2))
z = st.where(soft_cond, x, y)
print("soft z:", z.round(decimals=2))

# ── Soft argmax + indexing ──────────────────────────────────────────────────

x = torch.tensor([0.1, 0.4, 0.8])

idx = torch.argmax(x)
print("\nHard index:", idx)
y = x[idx]
print("Hard indexed value:", y)

soft_idx = st.argmax(x)
print("Soft index:", soft_idx.round(decimals=3))
y = st.index_select(x, soft_idx, dim=0)
print("Soft indexed value:", y.round(decimals=3))

# ── Soft argsort + take_along_dim ───────────────────────────────────────────

x = torch.tensor([0.3, 1.0, -0.5])

ind = torch.argsort(x)
print("\nHard sort indices:", ind)
values = torch.take_along_dim(x, ind, dim=0)
print("Hard sorted values:", values)

soft_idx = st.argsort(x)
print("Soft sort indices:", soft_idx.round(decimals=3))
soft_values = st.take_along_dim(x, soft_idx)
print("Soft sorted values:", soft_values.round(decimals=3))
