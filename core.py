import numpy as np
from typing import Any, Callable


def _no_dupes(entries: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    """Keep the first (name, value) pair for each distinct object."""
    seen = set()
    return [(name, v) for name, v in entries if not (id(v) in seen or seen.add(id(v)))]


class Parameter:
    data: np.ndarray
    grad: np.ndarray          # same shape and dtype as data
    name: str

    def __init__(self, data: np.ndarray, name: str = "") -> None:
        self.name = name
        self.grad = np.zeros_like(data)
        self.data = data

    def zero_grad(self) -> None:
        self.grad.fill(0)

    def to(self, dtype: np.dtype) -> None:
        self.data = self.data.astype(dtype)
        self.grad = self.grad.astype(dtype)

    @property
    def shape(self) -> tuple[int, ...]: return self.data.shape


class Module:
    training: bool
    _buffer_names: list[str]

    def __init__(self) -> None:
        """Subclasses call super().__init__() first."""
        self.training = True
        self._buffer_names = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the output and cache whatever backward needs."""
        raise NotImplementedError(f"{type(self).__name__} must implement forward")

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Given dL/d(output), add dL/d(param) into each param.grad and return dL/d(input)."""
        raise NotImplementedError(f"{type(self).__name__} must implement backward")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def _named_children(self) -> list[tuple[str, "Module"]]:
        ret = []

        for name, mod in vars(self).items():
            if isinstance(mod, Module):
                ret.append((name, mod))
            elif isinstance(mod, (list, tuple)):
                ret.extend((str(i), m) for i, m in enumerate(mod) if isinstance(m, Module))

        return ret

    def children(self) -> list["Module"]:
        return list({id(mod): mod for _, mod in self._named_children()}.values())

    def named_parameters(self, prefix: str = "") -> list[tuple[str, Parameter]]:
        """Dotted names, e.g. ('0.weight', p). Order must be deterministic: optimizers and torch weight copying rely on it."""
        return _no_dupes(self._named_members(Module._own_parameters, prefix + "." if prefix else ""))

    def _own_parameters(self) -> list[tuple[str, Parameter]]:
        return [(name, v) for name, v in vars(self).items() if isinstance(v, Parameter)]

    def _own_buffers(self) -> list[tuple[str, np.ndarray]]:
        return [(name, getattr(self, name)) for name in self._buffer_names]

    def _named_members(self, own: Callable, prefix: str) -> list[tuple[str, Any]]:
        ret = [(prefix + name, v) for name, v in own(self)]
        for name, mod in self._named_children():
            ret += mod._named_members(own, f"{prefix}{name}.")

        return ret

    def parameters(self) -> list[Parameter]:
        return [p for _, p in self.named_parameters()]

    def register_buffer(self, name: str, value: np.ndarray) -> None:
        self._buffer_names.append(name)
        setattr(self, name, value)

    def named_buffers(self, prefix: str = "") -> list[tuple[str, np.ndarray]]:
        """Non-trainable state, e.g. BatchNorm running_mean / running_var."""
        return _no_dupes(self._named_members(Module._own_buffers, prefix + "." if prefix else ""))

    def zero_grad(self) -> None:
        for para in self.parameters():
            para.zero_grad()

    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for mod in self.children():
            mod.train(mode)

        return self

    def eval(self) -> "Module":
        return self.train(False)

    def to(self, dtype: np.dtype) -> "Module":
        """Integer state such as a step counter is left alone, as in torch."""
        dtype = np.dtype(dtype)
        if not np.issubdtype(dtype, np.inexact):
            raise TypeError(f"Module.to only accepts floating point or complex dtypes, but got {dtype}")

        for _, para in self._own_parameters():
            if np.issubdtype(para.data.dtype, np.inexact):
                para.to(dtype)

        for name, buf in self._own_buffers():
            if np.issubdtype(buf.dtype, np.inexact):
                setattr(self, name, buf.astype(dtype))

        for mod in self.children():
            mod.to(dtype)

        return self

    def _state_arrays(self) -> dict[str, np.ndarray]:
        """Duplicate names would silently lose state, hence the check."""
        def own_state(mod: "Module") -> list[tuple[str, np.ndarray]]:
            return [(name, p.data) for name, p in mod._own_parameters()] + mod._own_buffers()

        entries = self._named_members(own_state, "")

        arrays = dict(entries)
        if len(arrays) != len(entries):
            seen = set()
            dupes = sorted({n for n, _ in entries if n in seen or seen.add(n)})
            raise ValueError(f"duplicate state names: {dupes}")

        return arrays

    def state_dict(self) -> dict[str, np.ndarray]:
        """Names should line up with the equivalent torch model's state_dict keys."""
        return {name: arr.copy() for name, arr in self._state_arrays().items()}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        """Validates everything before copying, so a failed load leaves the module unchanged."""
        arrays = self._state_arrays()

        missing = [name for name in arrays if name not in state]
        if missing:
            raise RuntimeError(f"missing keys in state: {missing}")

        for name, arr in arrays.items():
            if np.shape(state[name]) != arr.shape:
                raise RuntimeError(f"shape mismatch for {name}: expected {arr.shape}, got {tuple(np.shape(state[name]))}")

        for name, arr in arrays.items():
            arr[...] = state[name]

class Sequential(Module):
    layers: list[Module]

    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer(x) 

        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        for layer in self.layers[::-1]:
            dout = layer.backward(dout)
        
        return dout

    def __getitem__(self, i: int) -> Module: return self.layers[i]
    def __len__(self) -> int: return len(self.layers)