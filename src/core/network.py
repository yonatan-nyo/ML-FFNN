"""Feed-Forward Neural Network (FFNN) — from-scratch implementation."""

from __future__ import annotations

import json
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .activations import Activation, Softmax
from .initializers import Initializer
from .layers import Dense
from .losses import (
    CategoricalCrossEntropy,
    Loss,
    get_loss,
)


class NeuralNetwork:
    """Configurable FFNN with forward/backward propagation, regularisation,
    plotting utilities, and save/load support.

    Parameters
    ----------
    layer_sizes : list[int]
        Number of neurons in each layer **including** input and output.
        e.g. [784, 128, 64, 10]
    activations : list[Activation | str]
        Activation for each layer transition (len = len(layer_sizes)-1).
    loss : Loss | str
        Loss function instance or name.
    initializer : Initializer | str | None
        Weight initialiser (applied to every layer).
    regularization : str | None
        ``'l1'``, ``'l2'``, or ``None``.
    reg_lambda : float
        Regularisation strength.
    l1_lambda : float | None
        Optional explicit L1 coefficient. If provided, takes precedence.
    l2_lambda : float | None
        Optional explicit L2 coefficient. If provided, takes precedence.
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
    ):
        assert len(activations) == len(layer_sizes) - 1, (
            "Need exactly len(layer_sizes)-1 activation functions."
        )

        # Resolve loss
        if isinstance(loss, str):
            loss = get_loss(loss)
        self.loss_fn: Loss = loss

        # Regularisation (supports legacy regularization/reg_lambda and
        # explicit l1_lambda/l2_lambda simultaneously).
        self.l1_lambda, self.l2_lambda = self._resolve_regularization(
            regularization,
            reg_lambda,
            l1_lambda,
            l2_lambda,
        )
        self._sync_regularization_aliases()

        # Reproducibility: create a root RNG from the seed, then derive
        # if this isnt initialized ipynb result bakal beda2 tiap run
        self._root_rng = np.random.default_rng(seed)

        # Build layers
        self.layers: list[Dense] = []
        for i in range(len(layer_sizes) - 1):
            act = activations[i]
            # Derive a deterministic child seed for this layer
            layer_seed = int(self._root_rng.integers(0, 2**31)) if seed is not None else None
            self.layers.append(
                Dense(
                    input_dim=layer_sizes[i],
                    output_dim=layer_sizes[i + 1],
                    activation=act,
                    initializer=initializer,
                    seed=layer_seed,
                )
            )

        self._layer_sizes = layer_sizes
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

    # ──────────────────────────────────────────────
    # Forward propagation
    # ──────────────────────────────────────────────
    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Run forward pass through all layers. X: (batch, features)"""
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Public inference (no grad needed)."""
        return self._forward(np.asarray(X, dtype=np.float64))

    # ──────────────────────────────────────────────
    # Backward propagation
    # ──────────────────────────────────────────────
    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Compute gradients for all layers via chain rule."""
        last_layer = self.layers[-1]

        # Special-case: Softmax + CategoricalCrossEntropy  ->  dL/dz = y_pred - y_true
        if isinstance(last_layer.activation, Softmax) and isinstance(
            self.loss_fn, CategoricalCrossEntropy
        ):
            batch_size = y_true.shape[0]
            d_z = (y_pred - y_true) / batch_size  # (batch, C)
            # Manually set gradients for the last layer
            last_layer.grad_weights = last_layer._input.T @ d_z
            last_layer.grad_bias = np.sum(d_z, axis=0, keepdims=True)
            d_out = d_z @ last_layer.weights.T
            # Propagate through remaining layers
            for layer in reversed(self.layers[:-1]):
                d_out = layer.backward(d_out)
        else:
            # General case
            d_out = self.loss_fn.backward(y_true, y_pred)  # dL/d(y_pred)
            for layer in reversed(self.layers):
                d_out = layer.backward(d_out)

        # Add regularisation gradients
        if self.l1_lambda > 0 or self.l2_lambda > 0:
            for layer in self.layers:
                if self.l1_lambda > 0:
                    layer.grad_weights += self.l1_lambda * np.sign(layer.weights)
                if self.l2_lambda > 0:
                    layer.grad_weights += self.l2_lambda * layer.weights

    # ──────────────────────────────────────────────
    # Weight update (gradient descent)
    # ──────────────────────────────────────────────
    def _reset_optimizer_state(self) -> None:
        self._adam_m_w = [np.zeros_like(layer.weights) for layer in self.layers]
        self._adam_v_w = [np.zeros_like(layer.weights) for layer in self.layers]
        self._adam_m_b = [np.zeros_like(layer.bias) for layer in self.layers]
        self._adam_v_b = [np.zeros_like(layer.bias) for layer in self.layers]
        self._adam_t = 0

    def _update_weights(
        self,
        lr: float,
        optimizer: str = "sgd",
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        if optimizer == "sgd":
            for layer in self.layers:
                layer.weights -= lr * layer.grad_weights
                layer.bias -= lr * layer.grad_bias
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
            self._adam_m_w[i] = beta1 * self._adam_m_w[i] + (1.0 - beta1) * layer.grad_weights
            self._adam_v_w[i] = beta2 * self._adam_v_w[i] + (1.0 - beta2) * (layer.grad_weights ** 2)
            self._adam_m_b[i] = beta1 * self._adam_m_b[i] + (1.0 - beta1) * layer.grad_bias
            self._adam_v_b[i] = beta2 * self._adam_v_b[i] + (1.0 - beta2) * (layer.grad_bias ** 2)

            m_hat_w = self._adam_m_w[i] / (1.0 - beta1 ** self._adam_t)
            v_hat_w = self._adam_v_w[i] / (1.0 - beta2 ** self._adam_t)
            m_hat_b = self._adam_m_b[i] / (1.0 - beta1 ** self._adam_t)
            v_hat_b = self._adam_v_b[i] / (1.0 - beta2 ** self._adam_t)

            layer.weights -= lr * m_hat_w / (np.sqrt(v_hat_w) + epsilon)
            layer.bias -= lr * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

    # ──────────────────────────────────────────────
    # Compute loss (with optional regularisation term)
    # ──────────────────────────────────────────────
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        base_loss = self.loss_fn.forward(y_true, y_pred)
        if self.l1_lambda > 0 or self.l2_lambda > 0:
            reg_term = 0.0
            for layer in self.layers:
                if self.l1_lambda > 0:
                    reg_term += self.l1_lambda * np.sum(np.abs(layer.weights))
                if self.l2_lambda > 0:
                    reg_term += self.l2_lambda * np.sum(layer.weights ** 2)
            base_loss += reg_term
        return base_loss

    # ──────────────────────────────────────────────
    # Training loop
    # ──────────────────────────────────────────────
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
        """Train the network.

        Returns
        -------
        history : dict with 'train_loss' and 'val_loss' lists.
        """
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

        n_samples = X_train.shape[0]
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            # Shuffle (using the network's own RNG for reproducibility)
            indices = self._root_rng.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # Mini-batch training
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward
                y_pred = self._forward(X_batch)
                # Backward
                self._backward(y_batch, y_pred)
                # Update
                self._update_weights(
                    learning_rate,
                    optimizer=optimizer,
                    beta1=adam_beta1,
                    beta2=adam_beta2,
                    epsilon=adam_epsilon,
                )

            # Epoch losses
            train_pred = self._forward(X_train)
            train_loss = self._compute_loss(y_train, train_pred)
            history["train_loss"].append(train_loss)

            if val_data is not None:
                val_pred = self._forward(X_val)
                val_loss = self._compute_loss(y_val, val_pred)
            else:
                val_loss = float("nan")
            history["val_loss"].append(val_loss)

            # Verbose
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

    # ──────────────────────────────────────────────
    # Plotting
    # ──────────────────────────────────────────────
    def plot_weight_distribution(self, layers: list[int] | None = None) -> None:
        """Plot histogram of weights for selected layers."""
        if layers is None:
            layers = list(range(len(self.layers)))
        fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4))
        if len(layers) == 1:
            axes = [axes]
        for ax, idx in zip(axes, layers):
            w = self.layers[idx].weights.flatten()
            ax.hist(w, bins=50, edgecolor="black", alpha=0.7)
            ax.set_title(f"Layer {idx} weights")
            ax.set_xlabel("Weight value")
            ax.set_ylabel("Count")
        plt.tight_layout()
        plt.show()

    def plot_gradient_distribution(self, layers: list[int] | None = None) -> None:
        """Plot histogram of weight gradients for selected layers."""
        if layers is None:
            layers = list(range(len(self.layers)))
        fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4))
        if len(layers) == 1:
            axes = [axes]
        for ax, idx in zip(axes, layers):
            g = self.layers[idx].grad_weights.flatten()
            ax.hist(g, bins=50, edgecolor="black", alpha=0.7, color="orange")
            ax.set_title(f"Layer {idx} weight gradients")
            ax.set_xlabel("Gradient value")
            ax.set_ylabel("Count")
        plt.tight_layout()
        plt.show()

    # ──────────────────────────────────────────────
    # Save / Load
    # ──────────────────────────────────────────────
    def save(self, path: str) -> None:
        """Persist model to a JSON file."""
        data: dict[str, Any] = {
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
    def load(cls, path: str) -> "NeuralNetwork":
        """Restore model from a JSON file."""
        with open(path) as f:
            data = json.load(f)

        layer_dicts = data["layers"]
        activations_list = [ld["activation"] for ld in layer_dicts]

        model = cls(
            layer_sizes=data["layer_sizes"],
            activations=activations_list,
            loss=data["loss"],
            regularization=data.get("regularization"),
            reg_lambda=data.get("reg_lambda", 0.0),
            l1_lambda=data.get("l1_lambda"),
            l2_lambda=data.get("l2_lambda"),
        )
        # Restore saved weights
        for layer, ld in zip(model.layers, layer_dicts):
            layer.weights = np.array(ld["weights"])
            layer.bias = np.array(ld["bias"])

        return model

    # ──────────────────────────────────────────────
    # Summary / repr
    # ──────────────────────────────────────────────
    def summary(self) -> str:
        lines = ["NeuralNetwork Summary", "=" * 55]
        total = 0
        for i, layer in enumerate(self.layers):
            lines.append(f"  Layer {i}: {layer}")
            total += layer.num_params()
        lines.append("-" * 55)
        lines.append(f"  Loss       : {self.loss_fn.name()}")
        if self.l1_lambda == 0.0 and self.l2_lambda == 0.0:
            reg_summary = "None"
        else:
            reg_summary = f"L1={self.l1_lambda:g}, L2={self.l2_lambda:g}"
        lines.append(f"  Regularization: {reg_summary}")
        lines.append(f"  Total params: {total}")
        lines.append("=" * 55)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

