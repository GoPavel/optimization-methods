from simplex_method import simplex_method
import numpy as np
import scipy.optimize as opt


def run_simplex(A, b, c):
    A = np.array(A)
    b = np.array(b)
    c = np.array(c)
    x, res = simplex_method(A, b, c) if simplex_method(A, b, c) is not None else (None, None)
    res_np = opt.linprog(c=-c, A_eq=A, b_eq=b, method='simplex')
    print(x)
    print(res)
    print(res_np)
    if not res_np.success:
        assert x is None
    else:
        assert abs(res - (-res_np.fun)) <= 1e-9


def test_var_1():
    A = [[1, 2, -1, 2, 4],
         [0, -1, 2, 1, 3],
         [1, -3, 2, 2, 0]]
    b = [1, 3, 4]
    c = [1, -3, 2, 1, 4]
    run_simplex(A, b, c)


def test_var_2():
    A = [[1, 3, 0, 2, 1],
         [2, -1, 1, 2, 3],
         [1, -1, 2, 1, 0]]
    b = [1, 2, 4]
    c = [-1, -3, 2, 1, 4]
    run_simplex(A, b, c)


def test_var_3():
    A = [[-1, 3, 0, 2, 1],
         [2, -1, 1, 2, 3],
         [1, -1, 2, 1, 0]]
    b = [1, 4, 5]
    c = [-1, 0, -2, 5, 4]
    run_simplex(A, b, c)


def test_var_4():
    A = [[2, 3, 1, 2, 1],
         [2, 1, -3, 2, 1],
         [2, 1, 2, 1, 0]]
    b = [1, 3, 1]
    c = [-1, 1, -2, 1, 5]
    run_simplex(A, b, c)


def test_var_5():
    A = [[2, 1, 3, 4],
         [1, -1, 2, 1],
         [0, 0, 1, 3]]
    b = [2, 4, 1]
    c = [-2, 3, 4, -1]
    run_simplex(A, b, c)


def test_var_6():
    A = [[2, 3, 1, 2],
         [2, -1, 2, 1],
         [1, 1, 0, -1]]
    b = [3, 4, 1]
    c = [-2, 3, -3, 3]
    run_simplex(A, b, c)


def test_var_7():
    A = [[2, 3, -1, 2],
         [1, 1, 1, 1],
         [2, -1, 0, 2]]
    b = [1, 1, 2]
    c = [-2, 3, 4, -1]
    run_simplex(A, b, c)


def test_var_8():
    A = [[2, 1, 3, 4],
         [2, -1, 2, 1],
         [0, 0, 1, 2]]
    b = [1, 2, 4]
    c = [-2, 3, 4, -1]
    run_simplex(A, b, c)


def test_var_9():
    A = [[1, 2, 3, 1, 2, 5],
         [2, -3, 1, 2, 1, 4]]
    b = [1, 2]
    c = [-2, 3, 4, -1, 2, 1]
    run_simplex(A, b, c)


def test_var_10():
    A = [[3, 2, 1, -3, 2, 1],
         [1, 1, 0, 0, 1, 1]]
    b = [3, 2]
    c = [-2, 3, 1, 2, 0, 1]
    run_simplex(A, b, c)


def test_var_11():
    A = [[1, 2, 3, 4, 5, 6],
         [2, 1, -3, 2, 1, -3]]
    b = [1, 4]
    c = [1, -1, 2, 3, 1, 0]
    run_simplex(A, b, c)


def test_var_12():
    A = [[2, 3, -1, 0, 2, 1],
         [2, 0, 3, 0, 1, 1]]
    b = [1, 2]
    c = [-2, 3, 4, -1, 2, 1]
    run_simplex(A, b, c)
