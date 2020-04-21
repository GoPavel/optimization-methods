from simplex_method import simplex_method
import numpy as np


def is_int(x, eps):
    return abs(x - int(x)) <= eps


def add_rule_A(A, i, sign):
    return np.vstack((A, [sign if i == j else 0 for j in range(A.shape[1])]))


def add_rule_b(b, x_i):
    return np.concatenate([b, [x_i]])


def get_simplex_result(A, b, c):
    res = simplex_method(A, b, c)
    if res is None:
        return None, None
    return res


def branch_and_bound_method(A, b, c, eps=1e-10):
    result, x = get_simplex_result(A, b, c)
    if result is None:
        return None, None

    for i, x_i in enumerate(x):
        if not is_int(x_i, eps):
            lower_bound = int(x_i)
            lower_res, lower_x = branch_and_bound_method(add_rule_A(A, i, 1), add_rule_b(b, lower_bound), c)

            upper_res, upper_x = branch_and_bound_method(add_rule_A(A, i, -1), add_rule_b(b, lower_bound + 1), c)

            if lower_res is None:
                return upper_res, upper_x
            elif upper_res is None:
                return lower_res, lower_x

            if lower_res >= upper_res:
                return lower_res, lower_x
            else:
                return upper_res, upper_x
    return result, x
