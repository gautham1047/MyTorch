from core import Parameter, Module
import numpy as np

class Linear(Module):
    weight: Parameter   # (out_features, in_features)
    bias: Parameter     # (out_features,) if bias else None

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 init: str = "he", rng: np.random.Generator | None = None) -> None:
        super().__init__()
        rng = rng if rng is not None else np.random.default_rng()
        if init == "he":
            var = 2 / in_features
        elif init == "xavier":
            var = 2 / (in_features + out_features)
        else:
            raise ValueError(f"unknown init {init!r}, expected 'he' or 'xavier'")
        self.weight = Parameter(rng.standard_normal((out_features, in_features)) * np.sqrt(var))
        self.bias = Parameter(np.zeros(out_features)) if bias else None
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x (N, in) -> (N, out)."""
        self.x = x
        ret = x @ self.weight.data.T
        if self.bias is not None:
            ret = ret + self.bias.data

        return ret

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.x is None:
            raise Exception('cannot run backward pass before forward pass')
        
        self.weight.grad += dout.T @ self.x

        if self.bias is not None:
            self.bias.grad += dout.sum(axis=0)
        
        return dout @ self.weight.data
    
class ReLU(Module):
    def forward(self, x: np.ndarray) -> np.ndarray:
        mask = x.real > 0
        self.mask = mask
        return np.where(mask, x, 0)
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.mask
    
class Dropout(Module):
    p: float          # drop chance
    frozen: bool      # reuse the last mask instead of resampling

    def __init__(self, p: float = 0.5, rng: np.random.Generator | None = None) -> None: 
        super().__init__()

        self.p = p
        self.rng = rng if rng is not None else np.random.default_rng()
        self.frozen = False
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            if not self.frozen or self.mask is None:
                # cast so a float32 input stays float32
                self.mask = ((self.rng.random(x.shape) >= self.p) / (1 - self.p)).astype(x.real.dtype)
            return x * self.mask
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if not self.training:
            return dout
        return dout * self.mask

    def freeze_mask(self, frozen: bool = True) -> None:
        """Gradient checks call forward many times; a resampled mask would make f non-deterministic."""
        self.frozen = frozen

class BatchNorm1d(Module):
    weight: Parameter         # gamma, (D,), init ones
    bias: Parameter           # beta, (D,), init zeros
    running_mean: np.ndarray  # buffer, init zeros
    running_var: np.ndarray   # buffer, init ones

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()

        self.weight = Parameter(np.ones(num_features))
        self.bias = Parameter(np.zeros(num_features))
        self.register_buffer("running_mean", np.zeros(num_features))
        self.register_buffer("running_var", np.ones(num_features))
        # not used in the math; present so state_dict keys match torch's BatchNorm
        self.register_buffer("num_batches_tracked", np.array(0, dtype=np.int64))

        self.momentum = momentum
        self.eps = eps
        self.xhat = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            N = x.shape[0]
            if N < 2:
                raise ValueError("BatchNorm needs more than 1 value per channel in training mode")
            mu = x.mean(axis=0)
            var = ((x - mu) ** 2).mean(axis=0)
            m = self.momentum
            self.running_mean = (1-m) * self.running_mean + m * mu
            self.running_var =  (1-m) * self.running_var  + m * var * N / (N-1)
            self.num_batches_tracked += 1

            self.fwd_train = True # for backprop
        else:
            mu, var = self.running_mean, self.running_var
            self.fwd_train = False # for backprop

        self.inv_std = 1 / np.sqrt(var + self.eps)
        self.xhat = (x - mu) * self.inv_std # normalize

        return self.weight.data * self.xhat + self.bias.data # shift so that relu doesnt clip half of the data after normalizing; gamma beta learned

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.xhat is None:
            raise Exception('cannot run backward pass before forward pass')

        self.weight.grad += (dout * self.xhat).sum(0)
        self.bias.grad  += dout.sum(0)

        N = self.xhat.shape[0]
        dxhat = dout * self.weight.data

        if self.fwd_train:
            dx = (self.inv_std / N) * (N * dxhat
                                    - dxhat.sum(axis=0)
                                    - self.xhat * (dxhat * self.xhat).sum(axis=0))
        else:
            dx = dxhat * self.inv_std

        return dx

class BatchNorm2d(BatchNorm1d):
    def __init__(self, num_channels: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__(num_channels, eps, momentum)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x (N, C, H, W)."""
        N, C, H, W = x.shape
        x2d = np.einsum('nchw->nhwc', x).reshape(-1, C)        # (N*H*W, C)
        out = super().forward(x2d)
        return np.einsum('nhwc->nchw', out.reshape(N, H, W, C))   # (N, C, H, W)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        N, C, H, W = dout.shape
        x2d = np.einsum('nchw->nhwc', dout).reshape(-1, C)        # (N*H*W, C)
        out = super().backward(x2d)
        return np.einsum('nhwc->nchw', out.reshape(N, H, W, C))   # (N, C, H, W)

class Flatten(Module):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout.reshape(self.shape)
    
def conv_output(in_size: int, k: int, stride: int, pad: int) -> int: 
    return (in_size + 2 * pad - k) // stride + 1

def im2col(x: np.ndarray, kh: int, kw: int, stride: int, pad: int) -> tuple[np.ndarray, int, int]:
    N, C, H, W = x.shape
    out_h = conv_output(H, kh, stride, pad)
    out_w = conv_output(W, kw, stride, pad)
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"kernel ({kh}, {kw}) is larger than padded input ({H + 2*pad}, {W + 2*pad})")

    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    cols = np.empty((N, C, kh, kw, out_h, out_w), dtype=x.dtype)

    # loop over kernel offsets only; each slice grabs that offset for every output position at once
    for i in range(kh):
        i_end = i + stride * out_h
        for j in range(kw):
            j_end = j + stride * out_w
            cols[:, :, i, j] = x_pad[:, :, i:i_end:stride, j:j_end:stride]

    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, C * kh * kw)
    return cols, out_h, out_w

def col2im(cols: np.ndarray, x_shape: tuple[int, int, int, int], kh: int, kw: int,
           stride: int, pad: int) -> np.ndarray:
    N, C, H, W = x_shape
    out_h = conv_output(H, kh, stride, pad)
    out_w = conv_output(W, kw, stride, pad)

    cols = cols.reshape(N, out_h, out_w, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)  # (N, C, kh, kw, oh, ow)
    x_pad = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=cols.dtype)

    # within one (i, j) offset the strided slice never hits a pixel twice, so += is safe
    for i in range(kh):
        i_end = i + stride * out_h
        for j in range(kw):
            j_end = j + stride * out_w
            x_pad[:, :, i:i_end:stride, j:j_end:stride] += cols[:, :, i, j]

    return x_pad[:, :, pad:pad + H, pad:pad + W]

class Conv2d(Module):
    weight: Parameter   # (out_channels, in_channels, kh, kw)
    bias: Parameter     # (out_channels,) if bias else None

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int],
                 stride: int = 1, padding: int = 0, bias: bool = True,
                 rng: np.random.Generator | None = None) -> None:
        super().__init__()
        rng = rng if rng is not None else np.random.default_rng()
        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        var = 2 / (in_channels * kh * kw)
        self.weight = Parameter(rng.standard_normal((out_channels, in_channels, kh, kw)) * np.sqrt(var))
        self.bias = Parameter(np.zeros(out_channels)) if bias else None
        self.kh, self.kw = kh, kw
        self.stride = stride
        self.padding = padding
        self.cols = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x (N, C, H, W) -> (N, F, out_h, out_w)."""
        N = x.shape[0]
        F = self.weight.shape[0]
        cols, out_h, out_w = im2col(x, self.kh, self.kw, self.stride, self.padding)
        self.cols, self.x_shape = cols, x.shape

        out = cols @ self.weight.data.reshape(F, -1).T          # (N*oh*ow, F)
        if self.bias is not None:
            out = out + self.bias.data

        return out.reshape(N, out_h, out_w, F).transpose(0, 3, 1, 2)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.cols is None:
            raise Exception('cannot run backward pass before forward pass')

        F = dout.shape[1]
        dout2d = dout.transpose(0, 2, 3, 1).reshape(-1, F)      # (N*oh*ow, F)

        self.weight.grad += (dout2d.T @ self.cols).reshape(self.weight.shape)
        if self.bias is not None:
            self.bias.grad += dout2d.sum(axis=0)

        dcols = dout2d @ self.weight.data.reshape(F, -1)         # (N*oh*ow, C*kh*kw)
        return col2im(dcols, self.x_shape, self.kh, self.kw, self.stride, self.padding)
    
class MaxPool2d(Module):
    def __init__(self, kernel_size: int = 2, stride: int | None = None) -> None:
        super().__init__()
        self.k = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.idx = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x (N, C, H, W) -> (N, C, out_h, out_w)."""
        N, C, H, W = x.shape
        cols, out_h, out_w = im2col(x.reshape(N * C, 1, H, W), self.k, self.k, self.stride, 0)

        self.idx = cols.real.argmax(axis=1)
        self.x_shape = x.shape
        self.cols_shape = cols.shape
        self.dtype = cols.dtype

        out = cols[np.arange(cols.shape[0]), self.idx]
        return out.reshape(N, C, out_h, out_w)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Overlapping windows (stride < k) accumulate correctly through col2im."""
        if self.idx is None:
            raise Exception('cannot run backward pass before forward pass')

        N, C, H, W = self.x_shape
        dcols = np.zeros(self.cols_shape, dtype=np.result_type(self.dtype, dout.dtype))
        dcols[np.arange(dcols.shape[0]), self.idx] = dout.ravel()

        dx = col2im(dcols, (N * C, 1, H, W), self.k, self.k, self.stride, 0)
        return dx.reshape(N, C, H, W)
