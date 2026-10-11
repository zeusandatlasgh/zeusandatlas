import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def _to_tensor(a):
    """Convert list / numpy / pandas to a 1D float32 tensor."""
    return torch.tensor(np.asarray(a, dtype=np.float32)).flatten()


class LinearRegression(nn.Module):
    # (a) constructor
    def __init__(self, learning_rate=0.01, epochs=100):
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs

        # nn.Parameter -> tracked by autograd and returned by .parameters()
        self.w0 = nn.Parameter(torch.zeros(1))
        self.w1 = nn.Parameter(torch.zeros(1))

        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # storage for intermediate values during training
        self.history = {"w0": [], "w1": [], "loss": []}

    # (b) linear model
    def forward(self, x):
        return self.w0 + self.w1 * x

    # (c) training + R^2 on test data
    def fit(self, X_train, y_train, X_test, y_test):
        X_train, y_train = _to_tensor(X_train), _to_tensor(y_train)
        X_test, y_test = _to_tensor(X_test), _to_tensor(y_test)

        # keep training data for analysis_plot
        self.X_train, self.y_train = X_train, y_train

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()            # clear old gradients
            y_pred = self.forward(X_train)        # forward pass
            loss = self.loss_fn(y_pred, y_train)  # MSE
            loss.backward()                       # compute gradients
            self.optimizer.step()                 # update w0, w1

            self.history["w0"].append(self.w0.item())
            self.history["w1"].append(self.w1.item())
            self.history["loss"].append(loss.item())

        # R^2 on the test set
        with torch.no_grad():
            y_hat = self.forward(X_test)
            ss_res = torch.sum((y_test - y_hat) ** 2)
            ss_tot = torch.sum((y_test - y_test.mean()) ** 2)
            self.r2 = (1 - ss_res / ss_tot).item()

        print(f"w0 = {self.w0.item():.4f}, w1 = {self.w1.item():.4f}")
        print(f"Test R^2 = {self.r2:.4f}")

    # (d) prediction
    def predict(self, x):
        with torch.no_grad():
            return self.forward(_to_tensor(x)).numpy()

    # (e) diagnostic plots
    def analysis_plot(self):
        fig, ax = plt.subplots(2, 2, figsize=(11, 8))
        epochs = range(1, len(self.history["loss"]) + 1)

        # data + fitted line
        x = self.X_train.numpy()
        ax[0, 0].scatter(x, self.y_train.numpy(), alpha=0.6, label="Data")
        xs = np.linspace(x.min(), x.max(), 100)
        ax[0, 0].plot(xs, self.predict(xs), "r", label="Fitted line")
        ax[0, 0].set(title="Data and fit", xlabel="BCR", ylabel="AnnualProduction")
        ax[0, 0].legend()

        ax[0, 1].plot(epochs, self.history["w0"])
        ax[0, 1].set(title="w0 vs epoch", xlabel="Epoch", ylabel="w0")

        ax[1, 0].plot(epochs, self.history["w1"])
        ax[1, 0].set(title="w1 vs epoch", xlabel="Epoch", ylabel="w1")

        ax[1, 1].plot(epochs, self.history["loss"])
        ax[1, 1].set(title="Loss vs epoch", xlabel="Epoch", ylabel="MSE")

        fig.tight_layout()
        plt.show()
