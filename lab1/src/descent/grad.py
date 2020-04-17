from typing import Callable, Iterator, Tuple
import numpy as np
import one_dim_search.linear as lin
from more_itertools import last


def constant_step_chooser(f, x_k: np.ndarray, cur_grad: np.ndarray):
    return 1e-3


def generic_step_chooser(one_dim_search: Callable):
    def step_chooser(f, x_k, cur_grad):
        phi = lambda h: f(x_k - h * cur_grad)
        l, r = lin.search(0, 0.01, f=phi, eps=1e-3, multiplier=2)
        l, r = last(one_dim_search(l, r, f=phi, eps=1e-5))
        return (l + r) / 2

    return step_chooser


def gradient_descent_iter(*, f: Callable[[np.ndarray], float], f_grad: Callable[[np.ndarray], np.ndarray], eps: float,
                          start: np.ndarray, step_chooser=constant_step_chooser) -> Iterator[Tuple[int, np.ndarray]]:
    assert 0 <= 1 - eps < 1
    cur = start
    yield cur
    start_grad = f_grad(start)
    iter_cnt = 0
    while True:
        if iter_cnt and iter_cnt % 100000 == 0:
            print("gradient iter: %d" % iter_cnt)
            print(cur)
        cur_grad = f_grad(cur)

        if np.linalg.norm(cur_grad) ** 2 <= eps * np.linalg.norm(start_grad) ** 2:
            return

        step = step_chooser(f, cur, cur_grad)
        cur = cur - step * f_grad(cur)
        yield cur
        iter_cnt += 1


def gradient_descent(*args, **kwargs) -> Tuple[int, np.array]:
    return last(gradient_descent_iter(*args, **kwargs))


# f(x,y) = x^2 + y^2

if __name__ == '__main__':
    def f(p):
        return p[0] ** 2 + p[1] ** 2


    def f_grad(p):
        x, y = p[0], p[1]
        return np.array([2 * x, 2 * y])


    print(gradient_descent(f=f, f_grad=f_grad, eps=1e-5, start=np.array([3, 2])))
