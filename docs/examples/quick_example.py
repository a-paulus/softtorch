import matplotlib.pyplot as plt
import torch
import softtorch as st

torch.set_printoptions(precision=4, sci_mode=False)
torch.set_default_dtype(torch.float64)


# 1. Median regression
# Minimize the median absolute residual to be robust to outliers.

torch.manual_seed(0)
X = torch.randn(20, 3)
w_true = torch.tensor([1.0, -2.0, 0.5])
y = X @ w_true
y[0] = 1e6  # inject outlier


def median_regression_loss(w, X, y, mode="smooth"):
    residuals = y - X @ w
    return st.median(st.abs(residuals, mode=mode), mode=mode)


w = torch.zeros(3, requires_grad=True)
hard_loss = median_regression_loss(w, X, y, mode="hard")
print("=== 1. Robust median regression ===")
print("Hard grad:", torch.autograd.grad(hard_loss, w)[0])
soft_loss = median_regression_loss(w, X, y, mode="smooth")
print("Soft grad:", torch.autograd.grad(soft_loss, w)[0])

ws = []
w = torch.zeros(3)
for _ in range(50):
    ws.append(w.tolist())
    w.requires_grad_(True)
    loss = median_regression_loss(w, X, y)
    g = torch.autograd.grad(loss, w)[0]
    w = (w - 0.1 * g).detach()
print("Learned w:", w, " (true:", w_true, ")")


# 2. Top-k feature selection
# Discover which features of a trained model are important.
# 10 features total, only 3 informative — learn gating scores to find them.

n_features, k = 10, 3
torch.manual_seed(42)
X = torch.randn(100, n_features)
w_model = torch.tensor([0, 2.0, 0, -1.5, 0, 0, 0, 5.0, 0, 0])
y = X @ w_model + 0.1 * torch.randn(100)


def feature_selection_loss(g, X, y, w_model, mode="smooth"):
    _, soft_idx = st.topk(g, k=k, mode=mode, gated_grad=False)
    mask = soft_idx.sum(dim=0)
    y_pred = (X * mask) @ w_model
    return torch.mean(st.abs(y_pred - y))


g = torch.zeros(n_features, requires_grad=True)
print("\n=== 2. Top-k feature selection ===")
hard_loss = feature_selection_loss(g, X, y, w_model, mode="hard")
print("Hard grad:", torch.autograd.grad(hard_loss, g)[0] if hard_loss.requires_grad else torch.zeros_like(g))
soft_loss = feature_selection_loss(g, X, y, w_model, mode="smooth")
print("Soft grad:", torch.autograd.grad(soft_loss, g)[0])

gs = []
g = torch.zeros(n_features)
for _ in range(5):
    gs.append(g.tolist())
    g.requires_grad_(True)
    loss = feature_selection_loss(g, X, y, w_model)
    g_grad = torch.autograd.grad(loss, g)[0]
    g = (g - 0.001 * g_grad).detach()
print("Selected features:", torch.topk(g, k=k).indices)


# 3. Differentiable filter
# Learn a threshold that gates inputs.

x_filt = torch.tensor([0.2, 0.8, 0.5, 1.2, 0.1])
target_sum = 2.0  # sum of values above threshold should equal 2.0 (= 0.8 + 1.2)


def filter_loss(t, x, target, mode="smooth"):
    mask = st.greater(x, t, mode=mode)
    return (torch.sum(mask * x) - target) ** 2


t = torch.tensor(0.0, requires_grad=True)
print("\n=== 3. Differentiable threshold filtering ===")
hard_loss = filter_loss(t, x_filt, target_sum, mode="hard")
print("Hard grad:", torch.autograd.grad(hard_loss, t)[0] if hard_loss.requires_grad else torch.zeros_like(t))
soft_loss = filter_loss(t, x_filt, target_sum, mode="smooth")
print("Soft grad:", torch.autograd.grad(soft_loss, t)[0])

ts = []
t = torch.tensor(0.0)
for _ in range(20):
    ts.append(float(t))
    t.requires_grad_(True)
    loss = filter_loss(t, x_filt, target_sum)
    t_grad = torch.autograd.grad(loss, t)[0]
    t = (t - 0.1 * t_grad).detach()
print("Learned threshold:", t)


# 4. Differentiable rule-based classifier
# Learn decision boundaries: classify positive if ANY feature is in [lo, hi].
# The rule is true if any element of a feature is inside `[lo, hi]`.
x_rules = torch.tensor([[0.2, 0.8], [0.5, 0.3], [0.9, 0.1], [0.4, 0.7],
                         [0.1, 0.4], [0.2, 0.7], [0.4, 0.1], [0.4, 0.7],
                         [0.7, 0.29], [0.3, 0.3], [0.61, 0.25], [0.4, 0.6],
                         [0.0, 0.1], [0.5, 0.3], [0.4, 0.9], [0.1, 0.57]])
labels = torch.tensor([0.0, 1.0, 0.0, 1.0,
                        1.0, 0.0, 1.0, 1.0,
                        0.0, 1.0, 0.0, 1.0,
                        0.0, 1.0, 1.0, 1.0])


@st.st
def rule_loss(params, x, labels, mode="smooth"):
    lo, hi = params[0], params[1]
    above = st.greater(x, lo, mode=mode)
    below = st.less(x, hi, mode=mode)
    in_range = st.logical_and(above, below)
    preds = st.any(in_range, dim=-1)
    return ((preds - labels) ** 2).sum()


params = torch.tensor([0.0, 1.0], requires_grad=True)
print("\n=== 4. Differentiable rule-based classifier ===")
hard_loss = rule_loss(params, x_rules, labels, mode="hard")
print("Hard grad:", torch.autograd.grad(hard_loss, params)[0] if hard_loss.requires_grad else torch.zeros_like(params))
soft_loss = rule_loss(params, x_rules, labels, mode="smooth")
print("Soft grad:", torch.autograd.grad(soft_loss, params)[0])

params_hist = []
params = torch.tensor([0.0, 1.0])
for _ in range(20):
    params_hist.append(params.tolist())
    params.requires_grad_(True)
    loss = rule_loss(params, x_rules, labels)
    p_grad = torch.autograd.grad(loss, params)[0]
    params = (params - 0.01 * p_grad).detach()
print("Learned [lo, hi]:", params)


# ── Plot ─────────────────────────────────────────────────────────────────────
palette = ["#00bfff", "#e7a1e5", "#6dd1ac", "#e1be6a", "#368f80", "#889fd9", "#f4836d", "#cecece"]
informative = {i for i, v in enumerate(w_model) if v != 0}

fig, axes = plt.subplots(1, 4, figsize=(8, 2.5))

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Iteration", fontsize=7)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.margins(x=0)

ws = torch.tensor(ws)
for i in range(ws.shape[1]):
    axes[0].plot(ws[:, i], color=palette[i], label=f"w[{i}]")
    axes[0].axhline(w_true[i], color=palette[i], ls="--", alpha=0.3)
axes[0].set_title("Median regression", fontsize=8)
axes[0].legend(fontsize=6)

gs = torch.tensor(gs)
for i in range(gs.shape[1]):
    if i in informative:
        if i == 1:
            kw = {"lw": 1.5, "color": "#6dd1ac", "label": "Informative"}
        else:
            kw = {"lw": 1.5, "color": "#6dd1ac", "label": None}
    else:
        if i == 4:
            kw = {"alpha": 0.2, "color": "#889fd9", "label": "Uninformative"}
        else:
            kw = {"alpha": 0.2, "color": "#889fd9", "label": None}
    axes[1].plot(gs[:, i], **kw)
axes[1].set_title("Top-k feature selection", fontsize=8)
axes[1].legend(fontsize=6, title="Feature scores", title_fontsize=6)

axes[2].plot(ts, color=palette[0])
for xi in x_filt:
    axes[2].axhline(xi, ls="--", color=palette[-1], alpha=0.5)
axes[2].set_title("Threshold filtering", fontsize=8)

params_hist = torch.tensor(params_hist)
axes[3].plot(params_hist[:, 1], color=palette[0], label="higher bound")
axes[3].plot(params_hist[:, 0], color=palette[2], label="lower bound")
axes[3].axhline(0.3, ls="--", color=palette[2], alpha=0.5)
axes[3].axhline(0.6, ls="--", color=palette[0], alpha=0.5)
axes[3].set_title("Rule classifier", fontsize=8)
axes[3].legend(fontsize=6)

fig.tight_layout()
fig.savefig("docs/examples/quick_example_optimization.svg", bbox_inches="tight", transparent=True)
