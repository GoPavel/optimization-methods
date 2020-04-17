from typing import Callable, Iterator, Tuple

from more_itertools import last


def search_iter(l: float, r: float, *, f: Callable[[float], float], eps: float) -> Iterator[Tuple[float, float]]:
    delta = eps * 0.4
    yield l, r
    while (r - l) > eps:
        x1 = (l + r) / 2 - delta
        x2 = (l + r) / 2 + delta
        f1 = f(x1)
        f2 = f(x2)
        if f1 < f2:
            r = x2
        elif f1 > f2:
            l = x1
        else:
            l = x1
            r = x2
        yield l, r


def search(*args, **kwargs) -> Tuple[float, float]:
    return last(search_iter(*args, **kwargs))
