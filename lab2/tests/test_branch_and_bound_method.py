from branch_and_bound_method import branch_and_bound_method

import numpy as np
import scipy.optimize as opt


def test_branch_and_bound():
    A = [[2, 3, 1, 2],
         [2, -1, 2, 1],
         [1, 1, 0, -1]]
    b = [3, 4, 1]
    c = [-2, 3, -3, 3]

    A = np.array(A)
    b = np.array(b)
    c = np.array(c)
    res_np = opt.linprog(c=-c, A_eq=A, b_eq=b, method='simplex')
    res, x = branch_and_bound_method(A, b, c) if branch_and_bound_method(A, b, c) is not None else (None, None)
    print(x)
    print(res)
    print(res_np)
    if not res_np.success:
        assert x is None
    else:
        assert abs(res - (-res_np.fun)) <= 1e-9
