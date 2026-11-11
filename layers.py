from core import Parameter, Module, Sequential
import numpy as np

class Linear(Module):
    weight: Parameter   # (out_features, in_features)
    bias: Parameter     # (out_features,) if bias else None

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 init: str = "he", rng: np.random.Generator | None = None) -> None:
        """init='he': weight ~ N(0, 2/in). init='xavier': weight ~ N(0, 2/(in+out)). bias = 0."""
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
        """x (N, in) -> x @ weight.T + bias, shape (N, out). Cache x for backprop."""
        self.x = x
        ret = x @ self.weight.data.T
        if self.bias is not None:
            ret = ret + self.bias.data

        return ret

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """weight.grad += dout.T @ x;  bias.grad += dout.sum(0);  return dout @ weight."""
        if self.x is None:
            raise Exception('cannot run backward pass before forward pass')
        
        self.weight.grad += dout.T @ self.x

        if self.bias is not None:
            self.bias.grad += dout.sum(axis=0)
        
        return dout @ self.weight.data
    
class ReLU(Module):
    def forward(self, x: np.ndarray) -> np.ndarray:
        """mask = x.real > 0 (cached); return np.where(mask, x, 0)."""
        mask = x.real > 0
        self.mask = mask
        return np.where(mask, x, 0)
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """return dout * mask."""
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
        """Train: mask = (rng.random(x.shape) >= p) / (1 - p); return x * mask.
        If frozen, reuse the cached mask. Eval: return x unchanged (mask = 1)."""
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
        """x (N, C, H, W): transpose to (N*H*W, C), run BatchNorm1d.forward, reshape back.
        Statistics are per channel over N, H, W; the unbiased factor uses N*H*W."""
        N, C, H, W = x.shape
        x2d = np.einsum('nchw->nhwc', x).reshape(-1, C)        # (N*H*W, C)
        out = super().forward(x2d)
        return np.einsum('nhwc->nchw', out.reshape(N, H, W, C))   # (N, C, H, W)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Same reshape, BatchNorm1d.backward, reshape back."""
        N, C, H, W = dout.shape
        x2d = np.einsum('nchw->nhwc', dout).reshape(-1, C)        # (N*H*W, C)
        out = super().backward(x2d)
        return np.einsum('nhwc->nchw', out.reshape(N, H, W, C))   # (N, C, H, W)

class Flatten(Module):
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Cache x.shape; return x.reshape(N, -1)."""
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """return dout.reshape(cached_shape)."""
        return dout.reshape(self.shape)