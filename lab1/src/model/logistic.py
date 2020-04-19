import one_dim_search.golden as gold
from descent.grad import gradient_descent_iter, generic_step_chooser
from descent.newton import newton_descent_iter
import numpy as np
import pandas as pd
from scipy.special import expit
from functools import partial


def _add_pseudo_coord(X: np.ndarray):
    objects_count, _ = X.shape
    ones = np.ones((objects_count, 1))
    return np.hstack((X, ones))


class LogisticModel:
    def __init__(self, X: np.ndarray, y: np.ndarray, alpha: float, solver: str, eps: float, max_errors: int = 100):
        assert solver in {'gradient', 'newton'}
        self.alpha = alpha
        self.solver = solver
        self.eps = eps

        self.errors = 0
        self.max_errors = max_errors
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
            res = list(gradient_descent_iter(f=self._Q, f_grad=self._Q_grad, start=start_w,
                                      step_chooser=generic_step_chooser(gold.search), eps=self.eps))
            self.w = res[-1]
            self.num_steps = len(res)
        else:
            while True:
                try:
                    if self.errors >= self.max_errors:
                        self.w = start_w
                    else:
                        res = list(newton_descent_iter(f=self._Q, f_grad=self._Q_grad, f_hess=self._Q_hess, start=start_w,
                                                eps=self.eps))
                        self.w = res[-1]
                        self.num_steps = len(res)
                    return
                except ArithmeticError:
                    self.errors += 1
                    start_w = np.random.normal(loc=0., scale=1., size=self.features_count + 1)
            pass

    @property
    def num_errors(self):
        return self.errors

    def predict(self, X):
        if self.w is None:
            raise RuntimeError("Call fit function before predict")
        X = _add_pseudo_coord(X)
        return np.sign(np.matmul(X, self.w)).astype(int)


def read_dataset(path):
    data = pd.read_csv(path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].apply(lambda c: 1 if c == 'P' else -1).values
    return X, y
