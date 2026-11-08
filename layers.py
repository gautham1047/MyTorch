from core import Parameter, Module, Sequential
import numpy as np

class Linear(Module):
    W: Parameter   # (out_features, in_features)
    b: Parameter   # (out_features,) if bias else None

    def __init__(self, in_features: int, out_features: int, 
                 bias: bool = True, rng: np.random.Generator | None = None) -> None:
        """Note: b = 0."""
        super().__init__()
        rng = rng if rng is not None else np.random.default_rng()
        var = 2 / (in_features + out_features)
        self.W = Parameter(rng.standard_normal((out_features, in_features)) * np.sqrt(var))
        self.b = np.zeros(out_features) if bias else None
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x (N, in) -> x @ W.T + b, shape (N, out). Cache x for backprop."""
        self.x = x
        ret = x @ self.W.data.T
        if self.b is not None:
            ret = ret + self.b
        
        return ret

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """W.grad += dout.T @ x;  b.grad += dout.sum(0);  return dout @ W."""
        if self.x is None:
            raise Exception('cannot run backward pass before forward pass')
        
        self.W.grad = dout.T @ self.x

        if self.b is not None:
            self.b.grad += dout.sum(axis=0)
        
        return dout @ self.W.data
    
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
        self.rng = rng

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Train: mask = (rng.random(x.shape) >= p) / (1 - p); return x * mask.
        If frozen, reuse the cached mask. Eval: return x unchanged (mask = 1)."""
        if self.training:
            if not self.frozen:
                self.mask = (self.rng.random(x.shape) >= self.p) / (1 - self.p)
            return x * self.mask
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.mask 

    def freeze_mask(self, frozen: bool = True) -> None:
        """Gradient checks call forward many times; a resampled mask would make f non-deterministic."""
        self.frozen = frozen

class BatchNorm1d(Module):
    gamma: Parameter          # (D,), init ones
    beta: Parameter           # (D,), init zeros
    running_mean: np.ndarray  # buffer, init zeros
    running_var: np.ndarray   # buffer, init ones

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()

        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        


    def forward(self, x: np.ndarray) -> np.ndarray:
        """x (N, D).
        Train: mu = x.mean(0); var = ((x - mu)**2).mean(0)  (biased, used for normalizing)
               xhat = (x - mu) / sqrt(var + eps); out = gamma * xhat + beta
               running_mean = (1 - m) * running_mean + m * mu
               running_var  = (1 - m) * running_var  + m * var * N / (N - 1)   (unbiased, PyTorch convention)
        Eval:  normalize with running_mean / running_var.
        Cache xhat, inv_std = 1 / sqrt(var + eps)."""

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """gamma.grad += (dout * xhat).sum(0);  beta.grad += dout.sum(0)
        Train: compact formula below (derive it; don't backprop node by node).
        Eval:  dx = dout * gamma * inv_std_running."""