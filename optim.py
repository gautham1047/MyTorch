from core import Parameter
import numpy as np


class Optimizer:
    params: list[Parameter]
    lr: float
    state: dict[int, dict[str, np.ndarray]]   # keyed by id(param)

    def __init__(self, params: list[Parameter], lr: float) -> None:
        params = list(params)
        if not params:
            raise ValueError("optimizer got an empty parameter list")
        if lr < 0:
            raise ValueError(f"invalid learning rate {lr}")

        # drop repeats so a shared parameter is only stepped once
        self.params = list({id(p): p for p in params}.values())
        self.lr = lr
        self.state = {}

    def step(self) -> None:
        raise NotImplementedError(f"{type(self).__name__} must implement step")

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()


class SGD(Optimizer):
    def __init__(self, params: list[Parameter], lr: float = 1e-2, momentum: float = 0.0,
                 weight_decay: float = 0.0, nesterov: bool = False) -> None:
        super().__init__(params, lr)
        if nesterov and momentum <= 0:
            raise ValueError("nesterov momentum requires momentum > 0")

        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov

    def step(self) -> None:
        for p in self.params:
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data

            if self.momentum:
                state = self.state.setdefault(id(p), {})
                if "v" not in state:
                    state["v"] = g.copy()
                else:
                    state["v"] = self.momentum * state["v"] + g
                v = state["v"]
                g = g + self.momentum * v if self.nesterov else v

            p.data -= self.lr * g


class Adam(Optimizer):
    def __init__(self, params: list[Parameter], lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0) -> None:
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self) -> None:
        """Weight decay is L2 added to the gradient (Adam, not AdamW)."""
        b1, b2 = self.betas
        for p in self.params:
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data

            state = self.state.setdefault(id(p), {})
            if not state:
                state["t"] = 0
                state["m"] = np.zeros_like(p.data)
                state["v"] = np.zeros_like(p.data)

            state["t"] += 1
            t = state["t"]
            state["m"] = b1 * state["m"] + (1 - b1) * g
            state["v"] = b2 * state["v"] + (1 - b2) * g * g

            m_hat = state["m"] / (1 - b1 ** t)
            v_hat = state["v"] / (1 - b2 ** t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
