from simplex_method import simplex_method
import numpy as np


def is_int(x, eps):
    return abs(x - int(x)) <= eps


def add_rule_A(A, i, sign):
    return np.vstack((A, [sign if i == j else 0 for j in range(A.shape[1])]))


def add_rule_b(b, x_i):
    return b + [x_i]


def branch_and_bound_method(A, b, c, eps=1e-10):
    result = simplex_method(A, b, c)
    if result is None:
        return None

    for i, x_i in enumerate(result):
        if not is_int(x_i, eps):
            lower_bound = int(x_i)
            lower_res = branch_and_bound_method(add_rule_A(A, i, 1), add_rule_b(b, lower_bound), c)

            upper_res = branch_and_bound_method(add_rule_A(A, i, -1), add_rule_b(b, lower_bound + 1), c)

            if lower_res is None:
                return upper_res
            elif upper_res is None:
                return lower_res
            return max([upper_res, lower_res])
    return result
