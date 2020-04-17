from typing import Callable, Iterator, Tuple

from more_itertools import last

fib = [1, 1, 2]  # 3, 5, 8


def search_iter(l: float, r: float, *, f: Callable[[float], float], eps: float) -> Iterator[Tuple[float, float]]:
    # precalc fib
    global fib
    n = 0
    rel_len = (r - l) / eps
    while fib[n + 2] < rel_len:
        n += 1
        if len(fib) <= n + 2:
            fib.append(fib[n + 1] + fib[n])
    # search
    yield l, r
    #     3   2  3
    # l|----|--|----|r
    #      x1  x2
    x1 = fib[n] / fib[n + 2] * (r - l)
    x2 = fib[n + 1] / fib[n + 2] * (r - l)
    f1, f2 = f(x1), f(x2)
    for i in range(1, n + 1):
        if f2 < f1:
            l = x1
            x1 = x2
            x2 = l + fib[n + 1 - i] / fib[n + 2 - i] * (r - l)
            f1 = f2
            f2 = f(x2)
        else:
            r = x2
            x2 = x1
            x1 = l + fib[n - i] / fib[n + 2 - i] * (r - l)
            f2 = f1
            f1 = f(x1)
        yield l, r


def search(*args, **kwargs) -> Tuple[float, float]:
    return last(search_iter(*args, **kwargs))
