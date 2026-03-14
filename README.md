<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/logo/softtorch_logo_white_transparent.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/_static/logo/softtorch_logo_black_transparent.png">
    <img alt="SoftTorch logo" src="https://raw.githubusercontent.com/a-paulus/softtorch/main/docs/_static/logo/softtorch_logo_black_transparent.png" style="width:60%; max-width:320px; height:auto;">
  </picture>
</p>

# SoftTorch

[![PyPI version](https://img.shields.io/pypi/v/softtorch)](https://pypi.org/project/softtorch/)
[![Python version](https://img.shields.io/pypi/pyversions/softtorch)](https://pypi.org/project/softtorch/)
[![License](https://img.shields.io/pypi/l/softtorch)](https://github.com/a-paulus/softtorch/blob/main/LICENSE)

Looking for JAX? See [SoftJAX](https://github.com/a-paulus/softjax).

## In a nutshell

SoftTorch provides soft differentiable drop-in replacements for traditionally non-differentiable functions in [PyTorch](https://pytorch.org), including

- elementwise operators: `abs`, `relu`, `clamp`, `sign`, `round` and `heaviside`;
- tensor-valued operators: `(arg)max`, `(arg)min`, `(arg)quantile`, `(arg)median`, `(arg)sort`, `(arg)topk` and `rank`;
- comparison operators such as: `greater`, `eq` or `isclose`;
- logical operators such as: `logical_and`, `all` or `any`;
- functions for selection with indices such as: `where`, `take_along_dim` or `index_select`.

All operators offer multiple modes (controlling smoothness or boundedness of the relaxation) and adjustable softening strength.

All operators also support straight-through estimation, using the non-differentiable function in the forward pass and the soft relaxation in the backward pass.

SoftTorch functions are drop-in replacements for their non-differentiable PyTorch counterparts.
Special care is needed for functions operating on indices, as we relax discrete indices into distributions over indices, which modifies the shape of returned/accepted values.


## Installation
Requires Python 3.12+.
```
pip install softtorch
```


## Documentation

Available at https://a-paulus.github.io/softtorch/.


## Quick example
```python
import torch
import softtorch as st

x = torch.tensor([-0.2, -1.0, 0.3, 1.0])

# Elementwise functions
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
```
```
Torch absolute: tensor([0.2000, 1.0000, 0.3000, 1.0000])
SoftTorch absolute (hard mode): tensor([0.2000, 1.0000, 0.3000, 1.0000])
SoftTorch absolute (soft mode): tensor([0.1523, 0.9999, 0.2715, 0.9999])

Torch clamp: tensor([-0.2000, -0.5000,  0.3000,  0.5000])
SoftTorch clamp (hard mode): tensor([-0.2000, -0.5000,  0.3000,  0.5000])
SoftTorch clamp (soft mode): tensor([-0.1952, -0.4993,  0.2873,  0.4993])

Torch heaviside: tensor([0., 0., 1., 1.])
SoftTorch heaviside (hard mode): tensor([0., 0., 1., 1.])
SoftTorch heaviside (soft mode): tensor([0.1192, 0.0000, 0.9526, 1.0000])

Torch ReLU: tensor([0.0000, 0.0000, 0.3000, 1.0000])
SoftTorch ReLU (hard mode): tensor([0.0000, 0.0000, 0.3000, 1.0000])
SoftTorch ReLU (soft mode): tensor([0.0127, 0.0000, 0.3049, 1.0000])

Torch round: tensor([-0., -1.,  0.,  1.])
SoftTorch round (hard mode): tensor([-0., -1.,  0.,  1.])
SoftTorch round (soft mode): tensor([-0.0465, -1.0000,  0.1189,  1.0000])

Torch sign: tensor([-1., -1.,  1.,  1.])
SoftTorch sign (hard mode): tensor([-1., -1.,  1.,  1.])
SoftTorch sign (soft mode): tensor([-0.7616, -0.9999,  0.9051,  0.9999])
```

```python
# Tensor-valued operators
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
```
```
Torch max: tensor(1.)
SoftTorch max (hard mode): tensor(1.)
SoftTorch max (soft mode): tensor(0.8874)

Torch min: tensor(-1.)
SoftTorch min (hard mode): tensor(-1.)
SoftTorch min (soft mode): tensor(-0.8996)

Torch sort: tensor([-1.0000, -0.2000,  0.3000,  1.0000])
SoftTorch sort (hard mode): tensor([-1.0000, -0.2000,  0.3000,  1.0000])
SoftTorch sort (soft mode): tensor([-0.8792, -0.1641,  0.2767,  0.8738])

Torch quantile: tensor(-0.5200)
SoftTorch quantile (hard mode): tensor(-0.5200)
SoftTorch quantile (soft mode): tensor(-0.4501)

Torch median: tensor(-0.2000)
SoftTorch median (hard mode): tensor(-0.2000)
SoftTorch median (soft mode): tensor(-0.1641)

Torch topk: tensor([ 1.0000,  0.3000, -0.2000])
SoftTorch topk (hard mode): tensor([ 1.0000,  0.3000, -0.2000])
SoftTorch topk (soft mode): tensor([ 0.8738,  0.2767, -0.1641])

Torch rank: tensor([1, 0, 2, 3])
SoftTorch rank (hard mode): tensor([2., 1., 3., 4.])
SoftTorch rank (soft mode): tensor([1.9950, 1.0548, 3.0239, 3.9228])
```

```python
# Sort: sweep over methods
print("\nTorch sort:", torch.sort(x).values)
print("SoftTorch sort (softsort):", st.sort(x, method="softsort", softness=0.1).values)
print("SoftTorch sort (neuralsort):", st.sort(x, method="neuralsort", softness=0.1).values)
print("SoftTorch sort (fast_soft_sort):", st.sort(x, method="fast_soft_sort", softness=2.0).values)
print("SoftTorch sort (ot):", st.sort(x, method="ot", softness=0.1).values)
print("SoftTorch sort (sorting_network):", st.sort(x, method="sorting_network", softness=0.1).values)

# Sort: sweep over modes
print("\nTorch sort:", torch.sort(x).values)
for mode in ["hard", "smooth", "c0", "c1", "c2"]:
    print(f"SoftTorch sort ({mode}):", st.sort(x, softness=0.5, mode=mode).values)
```
```
Torch sort: tensor([-1.0000, -0.2000,  0.3000,  1.0000])
SoftTorch sort (softsort): tensor([-0.8996, -0.1705,  0.2847,  0.8874])
SoftTorch sort (neuralsort): tensor([-0.8792, -0.1641,  0.2767,  0.8738])
SoftTorch sort (fast_soft_sort): tensor([-0.7462, -0.1971,  0.2938,  0.8569])
SoftTorch sort (ot): tensor([-0.7324, -0.2396,  0.3286,  0.7434])
SoftTorch sort (sorting_network): tensor([-0.7999, -0.2672,  0.3847,  0.7863])

Torch sort: tensor([-1.0000, -0.2000,  0.3000,  1.0000])
SoftTorch sort (hard): tensor([-1.0000, -0.2000,  0.3000,  1.0000])
SoftTorch sort (smooth): tensor([-0.6057, -0.1997,  0.2729,  0.6281])
SoftTorch sort (c0): tensor([-1.0000, -0.6313,  0.6525,  0.9824])
SoftTorch sort (c1): tensor([-0.9982, -0.5432,  0.5814,  0.9837])
SoftTorch sort (c2): tensor([-0.9978, -0.4905,  0.5425,  0.9903])
```

```python
# Operators returning indices
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
```
```
Torch argmax: tensor(3)
SoftTorch argmax (hard mode): tensor([0., 0., 0., 1.])
SoftTorch argmax (soft mode): tensor([0.0215, 0.0022, 0.1176, 0.8586])

Torch argmin: tensor(1)
SoftTorch argmin (hard mode): tensor([0., 1., 0., 0.])
SoftTorch argmin (soft mode): tensor([0.0922, 0.8885, 0.0169, 0.0023])

Torch argquantile: Not implemented in standard PyTorch
SoftTorch argquantile (hard mode): tensor([0.6000, 0.4000, 0.0000, 0.0000])
SoftTorch argquantile (soft mode): tensor([0.5403, 0.3693, 0.0902, 0.0001])

Torch argmedian: tensor(0)
SoftTorch argmedian (hard mode): tensor([1., 0., 0., 0.])
SoftTorch argmedian (soft mode): tensor([0.8009, 0.0491, 0.1498, 0.0002])

Torch argsort: tensor([1, 0, 2, 3])
SoftTorch argsort (hard mode): tensor([[0., 1., 0., 0.],
        [1., 0., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]])
SoftTorch argsort (soft mode): tensor([[0.1494, 0.8496, 0.0009, 0.0000],
        [0.8009, 0.0491, 0.1498, 0.0002],
        [0.1418, 0.0001, 0.7899, 0.0681],
        [0.0011, 0.0000, 0.1784, 0.8205]])

Torch argtopk: tensor([3, 2, 0])
SoftTorch argtopk (hard mode): tensor([[0., 0., 0., 1.],
        [0., 0., 1., 0.],
        [1., 0., 0., 0.]])
SoftTorch argtopk (soft mode): tensor([[0.0011, 0.0000, 0.1784, 0.8205],
        [0.1418, 0.0001, 0.7899, 0.0681],
        [0.8009, 0.0491, 0.1498, 0.0002]])
```

```python
y = torch.tensor([0.2, -0.5, 0.5, -1.0])

# Comparison operators
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
```
```
Torch greater: tensor([False, False, False,  True])
SoftTorch greater (hard mode): tensor([0., 0., 0., 1.])
SoftTorch greater (soft mode): tensor([0.0180, 0.0067, 0.1192, 1.0000])

Torch greater equal: tensor([False, False, False,  True])
SoftTorch greater equal (hard mode): tensor([0., 0., 0., 1.])
SoftTorch greater equal (soft mode): tensor([0.0180, 0.0067, 0.1192, 1.0000])

Torch less: tensor([ True,  True,  True, False])
SoftTorch less (hard mode): tensor([1., 1., 1., 0.])
SoftTorch less (soft mode): tensor([0.9820, 0.9933, 0.8808, 0.0000])

Torch less equal: tensor([ True,  True,  True, False])
SoftTorch less equal (hard mode): tensor([1., 1., 1., 0.])
SoftTorch less equal (soft mode): tensor([0.9820, 0.9933, 0.8808, 0.0000])

Torch eq: tensor([False, False, False, False])
SoftTorch eq (hard mode): tensor([0., 0., 0., 0.])
SoftTorch eq (soft mode): tensor([0.0414, 0.0143, 0.3580, 0.0000])

Torch not equal: tensor([True, True, True, True])
SoftTorch not equal (hard mode): tensor([1., 1., 1., 1.])
SoftTorch not equal (soft mode): tensor([0.9586, 0.9857, 0.6420, 1.0000])

Torch isclose: tensor([False, False, False, False])
SoftTorch isclose (hard mode): tensor([0., 0., 0., 0.])
SoftTorch isclose (soft mode): tensor([0.0414, 0.0143, 0.3580, 0.0000])
```

```python
# Logical operators
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
```
```
Torch AND: tensor([False, False, False,  True])
SoftTorch AND: tensor([0.0700, 0.0600, 0.0800, 0.9000])

Torch OR: tensor([ True, False,  True,  True])
SoftTorch OR: tensor([0.7300, 0.4400, 0.8200, 1.0000])

Torch NOT: tensor([ True,  True, False, False])
SoftTorch NOT: tensor([0.9000, 0.8000, 0.2000, 0.0000])

Torch XOR: tensor([ True, False,  True, False])
SoftTorch XOR: tensor([0.6411, 0.3464, 0.7256, 0.1000])

Torch ALL: tensor(False)
SoftTorch ALL: tensor(0.0160)

Torch ANY: tensor(True)
SoftTorch ANY: tensor(1.)

Torch Where: tensor([ 0.2000, -0.5000,  0.3000,  1.0000])
SoftTorch Where: tensor([ 0.1600, -0.6000,  0.3400,  1.0000])
```

```python
# Straight-through operators: Use hard function on forward and soft on backward
print("Straight-through ReLU:", st.relu_st(x))
print("Straight-through sort:", st.sort_st(x).values)
print("Straight-through argtopk:", st.topk_st(x, k=3).indices)
print("Straight-through greater:", st.greater_st(x, y))
# And many more...
```
```
Straight-through ReLU: tensor([0.0000, 0.0000, 0.3000, 1.0000])
Straight-through sort: tensor([-1.0000, -0.2000,  0.3000,  1.0000])
Straight-through argtopk: tensor([[0., 0., 0., 1.],
        [0., 0., 1., 0.],
        [1., 0., 0., 0.]])
Straight-through greater: tensor([0., 0., 0., 1.])
```


## Citation

If this library helped your academic work, please consider citing:

```bibtex
@article{paulus2026softjax,
  title={{SoftJAX} \& {SoftTorch}: Empowering Automatic Differentiation Libraries with Informative Gradients},
  author={Paulus, Anselm and Geist, A.\ Ren\'e and Musil, V\'it and Hoffmann, Sebastian and Beker, Onur and Martius, Georg},
  journal={arXiv preprint},
  year={2026},
  eprint={2603.08824}
}
```

Also consider starring the project [on GitHub](https://github.com/a-paulus/softtorch)!

Special thanks and credit go to [Patrick Kidger](https://kidger.site) for the awesome [JAX repositories](https://github.com/patrick-kidger) that served as the basis for the documentation of this project.


## Feedback

This project is still relatively young, if you have any suggestions for improvement or other feedback, please [reach out](mailto:paulus.anselm@gmail.com) or raise a GitHub issue!


## See also

### Other libraries on differentiable programming

**Differentiable sorting, top-k and rank**
[DiffSort](https://github.com/Felix-Petersen/diffsort): Differentiable sorting networks in PyTorch.  
[DiffTopK](https://github.com/Felix-Petersen/difftopk): Differentiable top-k in PyTorch.  
[FastSoftSort](https://github.com/google-research/fast-soft-sort): Fast differentiable sorting and ranking in JAX.  
[Differentiable Top-k with Optimal Transport](https://gist.github.com/thomasahle/48e9b3f17ead6c3ef11325f25de3655e) in JAX.  
[SoftSort](https://github.com/sprillo/softsort): Differentiable argsort in PyTorch and TensorFlow.  

**Other**  
[DiffLogic](https://github.com/Felix-Petersen/difflogic): Differentiable logic gate networks in PyTorch.  
[SmoothOT](https://github.com/mblondel/smooth-ot): Smooth and Sparse Optimal Transport.  
[JaxOpt](https://github.com/google/jaxopt): Differentiable optimization in JAX.  

### Papers on differentiable algorithms
SoftTorch builds on / implements various different algorithms for e.g. differentiable `topk`, `sorting` and `rank`, including:

[Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application](https://arxiv.org/pdf/1309.1541)  
[Differentiable Ranks and Sorting using Optimal Transport](https://arxiv.org/pdf/1905.11885)  
[Differentiable Top-k with Optimal Transport](https://papers.nips.cc/paper/2020/file/ec24a54d62ce57ba93a531b460fa8d18-Paper.pdf)  
[SoftSort: A Continuous Relaxation for the argsort Operator](https://arxiv.org/pdf/2006.16038)  
[Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances](https://arxiv.org/abs/1306.0895)  
[Smooth and Sparse Optimal Transport](https://arxiv.org/abs/1710.06276)  
[Smooth Approximations of the Rounding Function](https://arxiv.org/pdf/2504.19026v1)  
[Fast Differentiable Sorting and Ranking](https://arxiv.org/pdf/2002.08871)  
[Differentiable Sorting Networks for Scalable Sorting and Ranking Supervision](https://arxiv.org/abs/2105.04019)  

Please check the [API Documentation](https://a-paulus.github.io/softtorch/api/softtorch_operators) for implementation details.