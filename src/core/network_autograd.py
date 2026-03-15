"""Autograd variant of the FFNN."""

from __future__ import annotations

import json
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .activations import Activation
from .initializers import Initializer
from .layers_autograd import AutogradDense
from .losses import Loss, get_loss
from .autograd import Tensor


class AutogradNeuralNetwork:
    """FFNN whose gradients are computed by the autograd engine.

    Same API as ``NeuralNetwork``.  During training, every parameter's
    ``.grad`` is populated automatically by ``loss.backward()`` — no
    hand-written derivative of any activation or loss is ever called.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activations: list[Activation | str],
        loss: Loss | str = "mse",
        initializer: Initializer | str | None = None,
        regularization: str | None = None,
        reg_lambda: float = 0.0,
        l1_lambda: float | None = None,
        l2_lambda: float | None = None,
        seed: int | None = None,
    ) -> None:
        assert len(activations) == len(layer_sizes) - 1, (
            "Need exactly len(layer_sizes)-1 activation functions."
        )

        if isinstance(loss, str):
            loss = get_loss(loss)
        self.loss_fn: Loss = loss

        self.l1_lambda, self.l2_lambda = self._resolve_regularization(
            regularization,
            reg_lambda,
            l1_lambda,
            l2_lambda,
        )
        self._sync_regularization_aliases()
        self._root_rng = np.random.default_rng(seed)
        self._layer_sizes = layer_sizes

        self.layers: list[AutogradDense] = []
        for i in range(len(layer_sizes) - 1):
            layer_seed = int(self._root_rng.integers(0, 2**31)) if seed is not None else None
            self.layers.append(
                AutogradDense(
                    input_dim=layer_sizes[i],
                    output_dim=layer_sizes[i + 1],
                    activation=activations[i],
                    initializer=initializer,
                    seed=layer_seed,
                )
            )

        self._adam_m_w: list[np.ndarray] | None = None
        self._adam_v_w: list[np.ndarray] | None = None
        self._adam_m_b: list[np.ndarray] | None = None
        self._adam_v_b: list[np.ndarray] | None = None
        self._adam_t: int = 0

    @staticmethod
    def _resolve_regularization(
        regularization: str | None,
        reg_lambda: float,
        l1_lambda: float | None,
        l2_lambda: float | None,
    ) -> tuple[float, float]:
        l1 = 0.0 if l1_lambda is None else float(l1_lambda)
        l2 = 0.0 if l2_lambda is None else float(l2_lambda)

        if l1_lambda is None and l2_lambda is None:
            if regularization == "l1":
                l1 = float(reg_lambda)
            elif regularization == "l2":
                l2 = float(reg_lambda)
        else:
            if regularization == "l1" and reg_lambda > 0 and l1_lambda is None:
                l1 = float(reg_lambda)
            if regularization == "l2" and reg_lambda > 0 and l2_lambda is None:
                l2 = float(reg_lambda)

        return max(0.0, l1), max(0.0, l2)

    def _sync_regularization_aliases(self) -> None:
        if self.l1_lambda > 0 and self.l2_lambda > 0:
            self.regularization = "l1_l2"
            self.reg_lambda = self.l1_lambda + self.l2_lambda
        elif self.l1_lambda > 0:
            self.regularization = "l1"
            self.reg_lambda = self.l1_lambda
        elif self.l2_lambda > 0:
            self.regularization = "l2"
            self.reg_lambda = self.l2_lambda
        else:
            self.regularization = None
            self.reg_lambda = 0.0

    # ── Forward ───────────────────────────────────────────────────────

    def _forward_tensor(self, X: Tensor) -> Tensor:
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._forward_tensor(Tensor(np.asarray(X, dtype=np.float64))).data

    # ── Training step ─────────────────────────────────────────────────

    def _reset_optimizer_state(self) -> None:
        self._adam_m_w = [np.zeros_like(layer.W.data) for layer in self.layers]
        self._adam_v_w = [np.zeros_like(layer.W.data) for layer in self.layers]
        self._adam_m_b = [np.zeros_like(layer.b.data) for layer in self.layers]
        self._adam_v_b = [np.zeros_like(layer.b.data) for layer in self.layers]
        self._adam_t = 0

    def _train_step(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray,
        lr: float,
        optimizer: str = "sgd",
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        for layer in self.layers:
            layer.zero_grad()

        y_pred_t = self._forward_tensor(Tensor(X_batch))
        loss_t = self.loss_fn.forward_tensor(Tensor(y_batch), y_pred_t)
        loss_t.backward()

        # Add regularisation directly to gradients (no graph needed)
        if self.l1_lambda > 0 or self.l2_lambda > 0:
            for layer in self.layers:
                if self.l1_lambda > 0:
                    layer.W.grad += self.l1_lambda * np.sign(layer.W.data)
                if self.l2_lambda > 0:
                    layer.W.grad += self.l2_lambda * layer.W.data

        if optimizer == "sgd":
            for layer in self.layers:
                layer.W.data -= lr * layer.W.grad
                layer.b.data -= lr * layer.b.grad
            return

        if optimizer != "adam":
            raise ValueError(f"Unsupported optimizer '{optimizer}'. Use 'sgd' or 'adam'.")

        if (
            self._adam_m_w is None
            or self._adam_v_w is None
            or self._adam_m_b is None
            or self._adam_v_b is None
        ):
            self._reset_optimizer_state()

        # Adam moments and bias-correction reference:
        # https://www.geeksforgeeks.org/deep-learning/adam-optimizer/
        self._adam_t += 1
        for i, layer in enumerate(self.layers):
            self._adam_m_w[i] = beta1 * self._adam_m_w[i] + (1.0 - beta1) * layer.W.grad
            self._adam_v_w[i] = beta2 * self._adam_v_w[i] + (1.0 - beta2) * (layer.W.grad ** 2)
            self._adam_m_b[i] = beta1 * self._adam_m_b[i] + (1.0 - beta1) * layer.b.grad
            self._adam_v_b[i] = beta2 * self._adam_v_b[i] + (1.0 - beta2) * (layer.b.grad ** 2)

            m_hat_w = self._adam_m_w[i] / (1.0 - beta1 ** self._adam_t)
            v_hat_w = self._adam_v_w[i] / (1.0 - beta2 ** self._adam_t)
            m_hat_b = self._adam_m_b[i] / (1.0 - beta1 ** self._adam_t)
            v_hat_b = self._adam_v_b[i] / (1.0 - beta2 ** self._adam_t)

            layer.W.data -= lr * m_hat_w / (np.sqrt(v_hat_w) + epsilon)
            layer.b.data -= lr * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

    # ── Loss (numpy scalar, for reporting) ────────────────────────────

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        base = self.loss_fn.forward(y_true, y_pred)
        if self.l1_lambda > 0 or self.l2_lambda > 0:
            reg = 0.0
            for layer in self.layers:
                if self.l1_lambda > 0:
                    reg += self.l1_lambda * float(np.sum(np.abs(layer.W.data)))
                if self.l2_lambda > 0:
                    reg += self.l2_lambda * float(np.sum(layer.W.data ** 2))
            base += reg
        return base

    # ── Fit ───────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        val_data: tuple[np.ndarray, np.ndarray] | None = None,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        optimizer: str = "sgd",
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        epochs: int = 100,
        verbose: int = 1,
    ) -> dict[str, list[float]]:
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        optimizer = optimizer.lower()
        if optimizer not in {"sgd", "adam"}:
            raise ValueError(f"Unsupported optimizer '{optimizer}'. Use 'sgd' or 'adam'.")
        if optimizer == "adam":
            self._reset_optimizer_state()
        else:
            self._adam_m_w = None
            self._adam_v_w = None
            self._adam_m_b = None
            self._adam_v_b = None
            self._adam_t = 0

        if val_data is not None:
            X_val = np.asarray(val_data[0], dtype=np.float64)
            y_val = np.asarray(val_data[1], dtype=np.float64)

        n = X_train.shape[0]
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            idx = self._root_rng.permutation(n)
            X_s, y_s = X_train[idx], y_train[idx]

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                self._train_step(
                    X_s[start:end],
                    y_s[start:end],
                    learning_rate,
                    optimizer=optimizer,
                    beta1=adam_beta1,
                    beta2=adam_beta2,
                    epsilon=adam_epsilon,
                )

            train_pred = self.predict(X_train)
            train_loss = self._compute_loss(y_train, train_pred)
            history["train_loss"].append(train_loss)

            val_loss = float("nan")
            if val_data is not None:
                val_loss = self._compute_loss(y_val, self.predict(X_val))
            history["val_loss"].append(val_loss)

            if verbose == 1:
                bar_len = 30
                filled = int(bar_len * epoch / epochs)
                bar = "=" * filled + ">" + "." * (bar_len - filled - 1)
                msg = (
                    f"\rEpoch {epoch}/{epochs} [{bar}] "
                    f"- train_loss: {train_loss:.6f}"
                )
                if val_data is not None:
                    msg += f" - val_loss: {val_loss:.6f}"
                sys.stdout.write(msg)
                sys.stdout.flush()

        if verbose == 1:
            sys.stdout.write("\n")
            sys.stdout.flush()

        return history

    # ── Plotting ──────────────────────────────────────────────────────

    def plot_weight_distribution(self, layers: list[int] | None = None) -> None:
        if layers is None:
            layers = list(range(len(self.layers)))
        fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4))
        if len(layers) == 1:
            axes = [axes]
        for ax, idx in zip(axes, layers):
            ax.hist(self.layers[idx].W.data.flatten(), bins=50, edgecolor="black", alpha=0.7)
            ax.set_title(f"Layer {idx} weights")
            ax.set_xlabel("Weight value")
            ax.set_ylabel("Count")
        plt.tight_layout()
        plt.show()

    def plot_gradient_distribution(self, layers: list[int] | None = None) -> None:
        if layers is None:
            layers = list(range(len(self.layers)))
        fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4))
        if len(layers) == 1:
            axes = [axes]
        for ax, idx in zip(axes, layers):
            ax.hist(
                self.layers[idx].W.grad.flatten(),
                bins=50, edgecolor="black", alpha=0.7, color="orange"
            )
            ax.set_title(f"Layer {idx} weight gradients")
            ax.set_xlabel("Gradient value")
            ax.set_ylabel("Count")
        plt.tight_layout()
        plt.show()

    # ── Save / Load ───────────────────────────────────────────────────

    def save(self, path: str) -> None:
        data: dict[str, Any] = {
            "type": "AutogradNeuralNetwork",
            "layer_sizes": self._layer_sizes,
            "loss": self.loss_fn.name(),
            "regularization": self.regularization,
            "reg_lambda": self.reg_lambda,
            "l1_lambda": self.l1_lambda,
            "l2_lambda": self.l2_lambda,
            "layers": [layer.to_dict() for layer in self.layers],
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "AutogradNeuralNetwork":
        with open(path) as f:
            data = json.load(f)
        activations = [d["activation"] for d in data["layers"]]
        net = cls(
            layer_sizes=data["layer_sizes"],
            activations=activations,
            loss=data["loss"],
            regularization=data.get("regularization"),
            reg_lambda=data.get("reg_lambda", 0.0),
            l1_lambda=data.get("l1_lambda"),
            l2_lambda=data.get("l2_lambda"),
        )
        net.layers = [AutogradDense.from_dict(d) for d in data["layers"]]
        return net

    # ── Summary / repr ────────────────────────────────────────────────

    def summary(self) -> str:
        lines = ["AutogradNeuralNetwork Summary", "=" * 55]
        total = 0
        for i, layer in enumerate(self.layers):
            lines.append(f"  Layer {i}: {layer}")
            total += layer.num_params()
        lines.append("-" * 55)
        lines.append(f"  Loss          : {self.loss_fn.name()}")
        if self.l1_lambda == 0.0 and self.l2_lambda == 0.0:
            reg_summary = "None"
        else:
            reg_summary = f"L1={self.l1_lambda:g}, L2={self.l2_lambda:g}"
        lines.append(f"  Regularization: {reg_summary}")
        lines.append(f"  Total params  : {total}")
        lines.append("=" * 55)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()
