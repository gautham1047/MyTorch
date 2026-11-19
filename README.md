# MyTorch

A small neural network library in pure NumPy, modeled on PyTorch's `nn` API. There is no autograd: every layer writes its own `forward` and `backward`, and the results are designed to match PyTorch's.

## Overview

| File | Contents |
| --- | --- |
| [core.py](core.py) | `Parameter`, `Module`, `Sequential` |
| [layers.py](layers.py) | Layers, plus the `im2col` / `col2im` helpers |
| [loss.py](loss.py) | `log_softmax`, `softmax`, `SoftmaxCrossEntropy` |
| [optim.py](optim.py) | `SGD`, `Adam` |

- **`Parameter`** holds a `data` array and a `grad` array of the same shape.
- **`Module`** is the base class for layers. `forward(x)` caches whatever the backward pass needs. `backward(dout)` takes dL/d(output), adds dL/d(param) into each `param.grad`, and returns dL/d(input).
- **Gradients accumulate** across `backward` calls, so call `zero_grad()` before each step.
- **Module utilities:** `Module` finds sub-modules and parameters by scanning attributes, so you don't register them. It also provides `train()`/`eval()`, `to(dtype)`, and `state_dict()`/`load_state_dict()`. State dict keys match the equivalent PyTorch model, so weights can be copied either way.
- **Complex inputs:** everything works with complex arrays, so gradients can be checked with the complex-step method.

```python
import numpy as np
from core import Sequential
from layers import Conv2d, BatchNorm2d, ReLU, MaxPool2d, Flatten, Linear
from loss import SoftmaxCrossEntropy
from optim import Adam

model = Sequential(
    Conv2d(1, 8, 3, padding=1), BatchNorm2d(8), ReLU(), MaxPool2d(2),   # (N, 8, 14, 14)
    Flatten(), Linear(8 * 14 * 14, 10),
)
loss_fn = SoftmaxCrossEntropy()
opt = Adam(model.parameters(), lr=1e-3)

x = np.random.randn(32, 1, 28, 28)
y = np.random.randint(0, 10, 32)

opt.zero_grad()
loss = loss_fn(model(x), y)
model.backward(loss_fn.backward())
opt.step()
```

## Layers

| Layer | Input → output | Parameters / buffers | Notes |
| --- | --- | --- | --- |
| `Linear(in, out, bias=True, init="he")` | `(N, in)` → `(N, out)` | `weight (out, in)`, `bias (out,)` | `init` is `"he"` or `"xavier"` |
| `Conv2d(C, F, kernel_size, stride=1, padding=0, bias=True)` | `(N, C, H, W)` → `(N, F, H', W')` | `weight (F, C, kh, kw)`, `bias (F,)` | He init; uses im2col (see [below](#convolutions)) |
| `MaxPool2d(kernel_size=2, stride=None)` | `(N, C, H, W)` → `(N, C, H', W')` | none | `stride` defaults to `kernel_size`; no padding |
| `BatchNorm1d(D, eps=1e-5, momentum=0.1)` | `(N, D)` → `(N, D)` | `weight`, `bias`; buffers `running_mean`, `running_var`, `num_batches_tracked` | Batch statistics while training, running statistics in eval |
| `BatchNorm2d(C, eps=1e-5, momentum=0.1)` | `(N, C, H, W)` → same | same as `BatchNorm1d` | Per-channel statistics over `N, H, W` |
| `ReLU()` | any → same | none | |
| `Dropout(p=0.5)` | any → same | none | Inverted dropout, so no change is needed at eval time; `freeze_mask()` keeps the mask fixed for gradient checks |
| `Flatten()` | `(N, ...)` → `(N, -1)` | none | |
| `Sequential(*layers)` | | its children's | Runs `backward` through the layers in reverse order |

## Losses

| Loss | Input | Output | Notes |
| --- | --- | --- | --- |
| `SoftmaxCrossEntropy()` | logits `(N, K)`, int labels `(N,)` | scalar mean loss | `backward(dloss=1.0)` returns dL/d(logits), `(N, K)` |
| `log_softmax(x, axis=-1)` / `softmax(x, axis=-1)` | any | same shape | Shifts by the maximum for numerical stability (real part only, so complex inputs work) |

## Optimizers

| Optimizer | Arguments | Notes |
| --- | --- | --- |
| `SGD` | `lr=1e-2, momentum=0, weight_decay=0, nesterov=False` | Momentum buffer starts as the first gradient, as in PyTorch |
| `Adam` | `lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0` | Bias-corrected; weight decay is L2 added to the gradient (Adam, not AdamW) |

Both have `step()` and `zero_grad()`. Per-parameter state is stored in `opt.state[id(param)]`.

## Convolutions

### Definition

`Conv2d` computes a cross-correlation, as PyTorch does: the kernel is not flipped. With input `x (N, C, H, W)`, weight `W (F, C, kh, kw)`, stride `s` and padding `p`, let `x̃` be `x` zero-padded by `p` on each spatial side. Then

$$
\text{out}[n, f, y, x] = b_f + \sum_{c=0}^{C-1} \sum_{i=0}^{k_h-1} \sum_{j=0}^{k_w-1} W[f, c, i, j]\; \tilde{x}[n, c,\; s y + i,\; s x + j]
$$

with output size (`conv_output`)

$$
H' = \left\lfloor \frac{H + 2p - k_h}{s} \right\rfloor + 1, \qquad W' = \left\lfloor \frac{W + 2p - k_w}{s} \right\rfloor + 1.
$$

### im2col: convolution as one matrix multiply

The triple sum is a dot product between the flattened kernel `W[f]` and one flattened window of `x̃`. `im2col` copies every window into a row of a matrix:

$$
X_{\text{col}} \in \mathbb{R}^{(N H' W') \times (C k_h k_w)}, \qquad X_{\text{col}}[(n, y, x),\, (c, i, j)] = \tilde{x}[n, c,\, s y + i,\, s x + j].
$$

Rows are ordered `(n, y, x)` and columns `(c, i, j)`. The column order matches `W.reshape(F, -1)`, so the whole layer becomes one matrix multiply:

$$
\text{Out}_{\text{2d}} = X_{\text{col}}\, W_{\text{mat}}^{\top} + b, \qquad W_{\text{mat}} = W.\text{reshape}(F,\, C k_h k_w),
$$

which gives an `(N·H'·W', F)` result that is reshaped and transposed back to `(N, F, H', W')`. This is the same computation as a `Linear` layer, with each window acting as one sample.

**How the windows are gathered.** `im2col` doesn't loop over output positions. It loops over the `kh·kw` kernel offsets `(i, j)`. For each offset, one strided slice `x̃[:, :, i : i + sH' : s, j : j + sW' : s]` gives that window element for every image, channel and output position at once. The cost is memory: with stride 1, `X_col` holds about `kh·kw` copies of the input.

### Backward pass

Write `G = dL/dOut_2d`, shape `(N·H'·W', F)`. Because the forward pass is `Linear` applied to `X_col`, the gradients are the `Linear` ones:

$$
\frac{\partial L}{\partial W_{\text{mat}}} = G^{\top} X_{\text{col}}, \qquad \frac{\partial L}{\partial b} = \sum_{\text{rows}} G, \qquad \frac{\partial L}{\partial X_{\text{col}}} = G\, W_{\text{mat}}.
$$

The weight gradient is reshaped back to `(F, C, kh, kw)`.

### col2im: getting dL/dx back from dL/dX_col

`im2col` is linear: it copies entries of `x`, so `vec(X_col) = P · vec(x)` for a 0/1 matrix `P`. By the chain rule,

$$
\frac{\partial L}{\partial x} = P^{\top} \frac{\partial L}{\partial X_{\text{col}}}.
$$

`col2im` applies `Pᵀ`. Each entry of `dX_col` is added back to the pixel it was copied from, and then the padding is cut off. When windows overlap (stride < kernel size), one pixel appears in several rows of `X_col`, so it receives the sum of all their gradients. `col2im` is therefore **not** the inverse of `im2col`: `col2im(im2col(x))` multiplies each pixel by the number of windows that contain it. It uses the same loop over kernel offsets as `im2col`, with `+=` on each strided slice. That is safe because a single offset's slice never hits the same pixel twice.

### MaxPool2d

Max pooling reuses the same machinery. `x` is reshaped to `(N·C, 1, H, W)` so each channel is pooled on its own, and `im2col` turns it into rows of `k·k` window values. The forward pass takes each row's maximum and saves its position (`argmax`). In the backward pass, each output gradient goes only to the element that was the maximum, written into a zero matrix shaped like `X_col`, and `col2im` maps it back to `(N, C, H, W)`. If windows overlap, the gradients add up just as they do for convolution.
