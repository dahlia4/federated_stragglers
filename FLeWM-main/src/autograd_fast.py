#!/usr/bin/env python

import numpy as np
import torch


def accuracy(Y, Yhat):
    """
    Function for computing accuracy

    Y is a vector of the true labels and Yhat is a vector of estimated 0/1 labels
    """

    acc = 0
    for y, yhat in zip(Y, Yhat):

        if y == yhat:
            acc += 1

    return acc / len(Y) * 100


class MLP:
    """
    Class for implementing a multilayer perceptron
    """

    def __init__(self, n_features, layer_sizes, learning_rate=0.01):
        """
        Constructor that initializes layers of appropriate width
        """

        layer_sizes = [n_features] + layer_sizes

        self.layers = []
        for i in range(len(layer_sizes) - 1):
            params = np.random.uniform(-1, 1, (layer_sizes[i] + 1, layer_sizes[i + 1]))
            self.layers.append(torch.tensor(params, requires_grad=True))

        self.learning_rate = learning_rate

    def __call__(self, x):
        """
        Impelementing the call () operator which simply calls each layer in the net
        sequentially using outputs of previous layers
        """

        # use ReLU activation for all layers except the last one
        # x = torch.tensor(x, dtype=torch.float64)
        out = x
        for layer in self.layers[0 : len(self.layers) - 1]:
            out = torch.hstack((out, torch.tensor([1.0], dtype=torch.float64)))
            out = out @ layer
            out = (out > 0) * out

        out = torch.hstack((out, torch.tensor([1.0], dtype=torch.float64)))
        out = out @ self.layers[-1]
        return 1 / (1 + torch.exp(-out))

    # def train(self, X, y):
    #     """
    #     Method that trains the network
    #     """
    #     self._zero_grad()
    #     pY = self(X)
    #     print(y, pY)
    #     loss = -(y * torch.log(pY) + (1 - y) * torch.log(1 - pY)).mean()
    #     loss.backward()

    #     return self._get_grads()

    def _zero_grad(self):
        """
        Method that sets the gradients of all parameters to zero
        """
        for layer in self.layers:
            layer.grad = torch.zeros(layer.shape, dtype=torch.float64)

    def _get_grads(self):
        """
        Method that returns the gradients of all parameters
        """
        return [layer.grad for layer in self.layers]

    def _set_grads(self, grads):
        """
        Method that sets the gradients of all parameters
        """
        for i, layer in enumerate(self.layers):
            layer.grad = grads[i]

    def _get_parameters(self):
        """
        Method that returns the parameters of all layers
        """
        return [layer.data for layer in self.layers]

    def _set_parameters(self, params):
        """
        Method that sets the parameters of all layers
        """
        for i, layer in enumerate(self.layers):
            layer.data = params[i]

    def step(self):
        """
        Take a step
        """
        for layer in self.layers:
            layer.data = layer.data - self.learning_rate * layer.grad

    def evaluate(self, Xmat, Y):
        """
        Method that evaluates the model on a dataset
        """
        with torch.no_grad():
            Xmat = torch.tensor(Xmat, dtype=torch.float64)
            Y = torch.tensor(Y, dtype=torch.float64)
            # Xmat = torch.from_numpy(Xmat).double().requires_grad_(True)
            # Y = torch.from_numpy(Y).double().requires_grad_(True)
            total_loss = 0
            for x, y in zip(Xmat, Y):
                preds = self(x)
                total_loss += -(
                    y * torch.log(preds) + (1 - y) * torch.log(1 - preds)
                ).mean()
        return total_loss, accuracy(Y, self.predict(Xmat))

    def train(self, Xmat, Y_train):
        """
        Method that trains the model
        """
        Xmat = torch.tensor(Xmat, dtype=torch.float64, requires_grad=True)
        Y_train = torch.tensor(Y_train, dtype=torch.float64, requires_grad=True)
        # Xmat = torch.from_numpy(Xmat).double().requires_grad_(True)
        # Y_train = torch.from_numpy(Y_train).double().requires_grad_(True)
        self._zero_grad()
        for x, y in zip(Xmat, Y_train):
            pY = torch.clamp(self(x), 1e-9, 1 - 1e-9)
            loss = -(y * torch.log(pY) + (1 - y) * torch.log(1 - pY)).mean()
            loss.backward()
        return self._get_grads()

    def predict(self, Xmat):
        with torch.no_grad():
            if isinstance(Xmat, np.ndarray):
                # Xmat = torch.from_numpy(Xmat).double()
                Xmat = torch.tensor(Xmat, dtype=torch.float64)
            return [int(self(x).data > 0.5) for x in Xmat]

    # def fit(self, Xmat_train, Y_train, Xmat_val=None, Y_val=None, max_epochs=100, verbose=False):

    #     # iterate over epochs
    #     for e in range(max_epochs):
    #         for (x, y) in zip(Xmat_train, Y_train):
    #             pY1 = self(torch.tensor(x))
    #             loss = -(y*torch.log(pY1) + (1-y)*torch.log(1-pY1)).mean()
    #             loss.backward()
    #             self.step()
    #             self._zero_grad()

    #         if verbose:

    #             train_acc = accuracy(Y_train, self.predict(Xmat_train))

    #             if Xmat_val is not None:
    #                 val_acc = accuracy(Y_val, self.predict(Xmat_val))
    #                 print(f"Epoch {e}: Training accuracy {train_acc:.0f}%, Validation accuracy {val_acc:.0f}%")
    #             else:
    #                 print(f"Epoch {e}: Training accuracy {train_acc:.0f}%")

    # def predict(self, Xmat):
    #     Xmat = torch.from_numpy(Xmat).double()
    #     with torch.no_grad():
    #         probas = self(Xmat)
    #         return np.array([int(p[0] > 0.5) for p in probas])
