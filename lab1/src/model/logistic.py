import one_dim_search.golden as gold
from descent.grad import gradient_descent, generic_step_chooser

import numpy as np
from scipy.special import expit
from functools import partial


def _add_pseudo_coord(X: np.ndarray):
    objects_count, _ = X.shape
    ones = np.ones((objects_count, 1))
    return np.hstack((X, ones))


class LogisticModel:
    def __init__(self, X: np.ndarray, y: np.ndarray, alpha: float, solver: str, eps: float):
        assert solver in {'gradient', 'newton'}
        self.alpha = alpha
        self.solver = solver
        self.eps = eps

        self.X = _add_pseudo_coord(X)
        self.y = y
        self.objects_count, self.features_count = X.shape
        assert y.shape == (self.objects_count,)

    def _Q(self, weights):
        predictions = np.matmul(self.X, weights)
        margins = predictions * self.y
        losses = np.logaddexp(0, -margins)
        return (np.sum(losses) / self.objects_count) + (np.sum(weights ** 2) * self.alpha / 2)

    def _Q_grad(self, weights):
        predictions = np.matmul(self.X, weights)
        margins = predictions * self.y
        b = expit(-margins)
        grad = -np.matmul(self.A, b) / self.objects_count
        return grad + self.alpha * weights

    def _Q_hess(self, weights):
        predictions = np.matmul(self.X, weights)
        margins = predictions * self.y
        A = np.transpose(self.X * expit(-margins).reshape((self.objects_count, 1)))
        B = self.X * expit(margins).reshape((self.objects_count, 1))
        hess = np.matmul(A, B) / self.objects_count
        return hess + self.alpha * np.eye(self.features_count + 1)

    def fit(self, start_w):
        # opt for Q_grad
        self.A = np.transpose(self.X * self.y.reshape((self.objects_count, 1)))

        if self.solver == 'gradient':
            _, self.w = gradient_descent(f=self._Q, f_grad=self._Q_grad, start=start_w,
                                         step_chooser=generic_step_chooser(gold.search), eps=self.eps)
        else:
            # uncomment when will be written
            #_, self.w = newton_descent(f=self._Q, f_grad=self._Q_grad, f_hess=self._Q_hess, start=start_w, eps=self.eps)
            pass

    def predict(self):
        if self.w is None:
            raise RuntimeError("Call fit function before predict")
        return np.sign(np.matmul(self.X, self.w)).astype(int)
