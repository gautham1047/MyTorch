import numpy as np


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shift by the real part only, so it stays complex-safe."""
    shifted = x - x.real.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.exp(log_softmax(x, axis))

class SoftmaxCrossEntropy:
    def __init__(self) -> None:
        self.probs = None
        self.y = None

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        """logits (N, K), y (N,) int class labels. Returns a scalar (complex under complex step)."""
        y = np.asarray(y)
        if logits.ndim != 2 or y.shape != (logits.shape[0],):
            raise ValueError(f"expected logits (N, K) and labels (N,), got {logits.shape} and {y.shape}")

        logp = log_softmax(logits, axis=1)
        self.probs = np.exp(logp)
        self.y = y
        return -logp[np.arange(len(y)), y].mean()

    def backward(self, dloss: float = 1.0) -> np.ndarray:
        if self.probs is None:
            raise Exception('cannot run backward pass before forward pass')

        N = len(self.y)
        dlogits = self.probs.copy()
        dlogits[np.arange(N), self.y] -= 1
        return dloss * dlogits / N

    def __call__(self, logits: np.ndarray, y: np.ndarray) -> float:
        return self.forward(logits, y)
