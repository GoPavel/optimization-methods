from typing import Callable, Iterator
import numpy as np
from more_itertools import last


def newton_descent_iter(*, f: Callable[[np.ndarray], float],
                        f_grad: Callable[[np.ndarray], np.ndarray],
                        f_hess: Callable[[np.ndarray], np.ndarray], eps: float,
                        start: np.ndarray, cho_mode=False) -> Iterator[np.ndarray]:
    assert 0 <= 1 - eps < 1
    cur = start
    yield cur
    start_hess = f_hess(start)
    iter_cnt = 0
    while True:
        if iter_cnt and iter_cnt % 100000 == 0:
            print("newton iter: %d" % iter_cnt)
            print(cur)
        grad = f_grad(cur)
        hess = f_hess(cur)
        hess_inv = np.linalg.inv(hess)

        delta = np.matmul(grad, hess_inv)
        if np.linalg.norm(delta) < eps:
            return

        cur = cur - delta
        yield cur
        iter_cnt += 1


def newton_descent(*args, **kwargs) -> np.array:
    return last(newton_descent_iter(*args, **kwargs))


# f(x,y) = x^2 + y^2

if __name__ == '__main__':
    def f(p):
        return p[0] ** 2 + p[1] ** 2


    def f_grad(p):
        x, y = p[0], p[1]
        return np.array([2 * x, 2 * y])

    def f_hess(p):
        return np.array([[2, 0], [0, 2]])

    print(newton_descent(f=f, f_grad=f_grad, f_hess=f_hess, eps=1e-5, start=np.array([3, 2])))
