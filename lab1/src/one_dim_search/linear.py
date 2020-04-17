from typing import Callable, Tuple


def search(start, delta, *, f: Callable[[float], float], eps: float, multiplier: float) -> Tuple[float, float]:
    assert delta > 0, "Delta must be greater than 0"
    assert multiplier > 1, "Multiplier must be grater that 1"

    if f(start) < f(start + delta):
        delta = -delta

    start_y = f(start)
    x = start + delta
    step = delta

    while f(x) <= start_y + eps:
        step = step * multiplier
        x = x + step

    if delta > 0:
        return start, x
    else:
        return x, start
