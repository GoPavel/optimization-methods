from typing import Callable, Iterator, Tuple
import numpy as np


def constant_step_chooser(f, x_k: np.ndarray, cur_grad: np.ndarray):
    # g(a) = f(x_k - a * f_grad(x_k))
    return 1e-3


def gradient_descent(*, f: Callable[[np.ndarray], float], f_grad: Callable[[np.ndarray], np.ndarray], eps: float,
                     start: np.ndarray, step_chooser=constant_step_chooser) -> Tuple[int, np.ndarray]:
    assert 0 <= 1 - eps < 1
    cur = start
    start_grad = f_grad(start)
    iter_cnt = 0
    while True:
        if iter_cnt and iter_cnt % 100000 == 0:
            print("gradient iter: %d" % iter_cnt)
            print(cur)
        cur_grad = f_grad(cur)

        if np.linalg.norm(cur_grad) ** 2 <= eps * np.linalg.norm(start_grad) ** 2:
            return iter_cnt, cur

        step = step_chooser(f, cur, cur_grad)
        cur = cur - step * f_grad(cur)
        iter_cnt += 1


# f(x,y) = x^2 + y^2

if __name__ == '__main__':

    def f(p):
        return p[0] ** 2 + p[1] ** 2


    def f_grad(p):
        x, y = p[0], p[1]
        return np.array([2 * x, 2 * y])


    print(gradient_descent(f=f, f_grad=f_grad, eps=1e-5, start=np.array([3, 2])))
