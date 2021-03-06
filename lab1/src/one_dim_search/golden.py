from typing import Callable, Iterator, Tuple
from math import sqrt

from more_itertools import last


def search_iter(l, r, *, f: Callable[[float], float], eps: float) -> Iterator[Tuple[float, float]]:
    phi = (1 + sqrt(5)) / 2

    # optimization to calculate only one f-value in each step
    x_l = l + (2 - phi) * (r - l)
    x_r = r - (2 - phi) * (r - l)

    f_l = f(x_l)
    f_r = f(x_r)

    yield l, r

    while (r - l) > eps:
        if f_l < f_r:
            r = x_r
            x_r = x_l
            f_r = f_l

            x_l = l + (2 - phi) * (r - l)
            f_l = f(x_l)
        elif f_l > f_r:
            l = x_l
            x_l = x_r
            f_l = f_r

            x_r = r - (2 - phi) * (r - l)
            f_r = f(x_r)
        else:
            break
        yield l, r


def search(*args, **kwargs) -> Tuple[float, float]:
    return last(search_iter(*args, **kwargs))
