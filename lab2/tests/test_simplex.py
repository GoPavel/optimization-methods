import attr
import pytest

from simplex_method import simplex_method
import numpy as np
import scipy.optimize as opt


@attr.s(frozen=True)
class Sample:
    A = attr.ib(converter=np.array)
    b = attr.ib(converter=np.array)
    c = attr.ib(converter=np.array)


samples = [
    Sample(
        A=[[1, 2, -1, 2, 4],
           [0, -1, 2, 1, 3],
           [1, -3, 2, 2, 0]],
        b=[1, 3, 4],
        c=[1, -3, 2, 1, 4],
    ), Sample(
        A=[[1, 3, 0, 2, 1],
           [2, -1, 1, 2, 3],
           [1, -1, 2, 1, 0]],
        b=[1, 2, 4],
        c=[-1, -3, 2, 1, 4],
    ), Sample(
        A=[[-1, 3, 0, 2, 1],
           [2, -1, 1, 2, 3],
           [1, -1, 2, 1, 0]],
        b=[1, 4, 5],
        c=[-1, 0, -2, 5, 4],
    ), Sample(
        A=[[2, 3, 1, 2, 1],
           [2, 1, -3, 2, 1],
           [2, 1, 2, 1, 0]],
        b=[1, 3, 1],
        c=[-1, 1, -2, 1, 5],
    ), Sample(
        A=[[2, 1, 3, 4],
           [1, -1, 2, 1],
           [0, 0, 1, 3]],
        b=[2, 4, 1],
        c=[-2, 3, 4, -1],
    ), Sample(
        A=[[2, 3, 1, 2],
           [2, -1, 2, 1],
           [1, 1, 0, -1]],
        b=[3, 4, 1],
        c=[-2, 3, -3, 3],
    ), Sample(
        A=[[2, 3, -1, 2],
           [1, 1, 1, 1],
           [2, -1, 0, 2]],
        b=[1, 1, 2],
        c=[-2, 3, 4, -1],
    ), Sample(
        A=[[2, 1, 3, 4],
           [2, -1, 2, 1],
           [0, 0, 1, 2]],
        b=[1, 2, 4],
        c=[-2, 3, 4, -1],
    ), Sample(
        A=[[1, 2, 3, 1, 2, 5],
           [2, -3, 1, 2, 1, 4]],
        b=[1, 2],
        c=[-2, 3, 4, -1, 2, 1],
    ), Sample(
        A=[[3, 2, 1, -3, 2, 1],
           [1, 1, 0, 0, 1, 1]],
        b=[3, 2],
        c=[-2, 3, 1, 2, 0, 1],
    ), Sample(
        A=[[1, 2, 3, 4, 5, 6],
           [2, 1, -3, 2, 1, -3]],
        b=[1, 4],
        c=[1, -1, 2, 3, 1, 0],
    ), Sample(
        A=[[2, 3, -1, 0, 2, 1],
           [2, 0, 3, 0, 1, 1]],
        b=[1, 2],
        c=[-2, 3, 4, -1, 2, 1],
    )
]


@pytest.mark.parametrize("s", samples)
def test_eq(s):
    A, b, c = attr.astuple(s)
    print(f'A:\n{A}\nb:{b}, c:{c}')
    t = simplex_method(A, b, c)
    res, x = t if t is not None else (None, None)
    res_np = opt.linprog(c=-c, A_eq=A, b_eq=b, method='simplex')
    print(f'x:{x}\nres:{res}\n{res_np}')
    if not res_np.success:
        assert x is None
    else:
        assert abs(res - (-res_np.fun)) <= 1e-9

@pytest.mark.parametrize("s", samples)
def test_leq(s):
    A, b, c = attr.astuple(s)
    print(f'A:\n{A}\nb:{b}, c:{c}')
    t = simplex_method(A, b, c, leq=True)
    res, x = t if t is not None else (None, None)
    res_np = opt.linprog(c=-c, A_ub=A, b_ub=b, method='simplex')
    print(f'x:{x}\nres:{res}\nActual:\n{res_np}')
    if not res_np.success:
        assert x is None
    else:
        assert abs(res - (-res_np.fun)) <= 1e-9
