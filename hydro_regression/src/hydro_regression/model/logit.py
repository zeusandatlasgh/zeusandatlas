"""Logistic regression trained with mini-batch SGD on cross-entropy, in PyTorch."""

import numpy as np
import torch


class LogisticRegression:
    """Logistic regression trained with mini-batch SGD on cross-entropy, in PyTorch."""

    def __init__(self, lr=0.1, epochs=30, batch_size=64, seed=0, verbose=True):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose
        self.w_ = None               # weights, shape (n_features,)
        self.b_ = None               # bias
        self.loss_history = []       # training cross-entropy after each epoch

    @staticmethod
    def _sigmoid(z):
        return torch.sigmoid(z)

    @staticmethod
    def _cross_entropy(p, y, eps=1e-12):
        p = p.clamp(eps, 1 - eps)
        return -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()

    def fit(self, X, y):
        X = torch.tensor(np.asarray(X, dtype=np.float64))
        y = torch.tensor(np.asarray(y, dtype=np.float64))
        n, d = X.shape

        w = torch.zeros(d, dtype=X.dtype, requires_grad=True)
        b = torch.zeros(1, dtype=X.dtype, requires_grad=True)
        opt = torch.optim.SGD([w, b], lr=self.lr)

        gen = torch.Generator().manual_seed(self.seed)
        self.loss_history = []

        for epoch in range(self.epochs):
            perm = torch.randperm(n, generator=gen)        # reshuffle every epoch
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb, yb = X[idx], y[idx]

                opt.zero_grad()
                p = self._sigmoid(xb @ w + b)
                loss = self._cross_entropy(p, yb)
                loss.backward()
                opt.step()

            with torch.no_grad():                          # full-data loss for a clean curve
                full = self._cross_entropy(self._sigmoid(X @ w + b), y).item()
            self.loss_history.append(full)
            if self.verbose and (epoch % 5 == 0 or epoch == self.epochs - 1):
                print(f"epoch {epoch:3d}  loss {full:.5f}")

        self.w_ = w.detach().numpy()
        self.b_ = b.detach().numpy()
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        z = X @ self.w_ + self.b_
        return 1 / (1 + np.exp(-z))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
