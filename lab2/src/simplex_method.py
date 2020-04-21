import math

import numpy as np

"""
Simple simplex method from: 
    http://www.itlab.unn.ru/uploads/opt/optBook1.pdf
"""

def gauss_step(A, r, s):
    n, m = A.shape
    qrs = A[r, s]
    for j in range(m):
        A[r, j] /= qrs
    for i in range(n):
        if r == i:
            continue
        for j in range(m):
            A[i, j] -= A[r, j] * A[i, s]

def core(A, B) -> bool:
    """
    A = ( c )
        (b|A)
    Presume: b[i] >= 0
    B - basis
    :return isSystem has bound solution
    """
    # assert np.all(A[:, [0]] >= 0)

    n, m = A.shape
    while True:
        s = None
        for j in range(1, n):
            if A[0, j] > 0:
                s = j
                break
        if s is None:
            return True

        r, t = None, math.inf  # TODO (Why min? Maybe use other way)
        for i in range(1, n):
            if A[i, s] > 0 and A[i, 0] / A[i, s] < t:
                r = i
                t = A[i, 0] / A[i, s]
        if r is None:
            return False

        # Gauss step
        gauss_step(A, r, s)

        B[r-1] = s


def simplex_method(A, b, c):
    """
    Presume:
    forall i, b[i] >= 0
    A, b, c : np.ndarray
    Ax = b
    cx -> max
    """

    m, n = A.shape

    # first phase
    ## intros fake variables
    A = np.hstack((A, np.eye(len(A))))

    ## intro new function (TODO: Why?)
    """
    f = max(-x[n+1] - ... -x[n+m])
    """
    # add top row (0,0,0,...-x[n+1], -x[n+2]... -x[n+m]) and left row (0, b[0], b[1]...b[n])
    z = np.concatenate((np.zeros(1), np.zeros(n), np.repeat(1, m)))
    Bz = np.arange(n + 1, n + m + 1)
    A = np.hstack((b.reshape((len(b), 1)), A))
    A = np.vstack((z.reshape((1, len(z))), A))

    # row[0] -= sum(row[1..m])
    for i in range(1, m+1):
        A[0] = A[0] - A[i]

    # for r, i in enumerate(range(n+1, n+m+1), start=1):
    #     gauss_step(A, r, i)


    # set invariant B_z
    for b_ind, s in enumerate(Bz):
        if n < s:
            r = None
            for i in range(1, m+1):
                if r is None and A[i, s] == 1:
                    r = i
                else:
                    assert A[i, s] == 0, "in column more that 1 not zero element"
            assert r is not None, "0-base vector"

            k = None
            for i in range(1, n+1):
                if A[r, i] != 0:
                    k = i
                    break
            if k is None:
                A = np.delete(A, r, 0)
            else:
                gauss_step(A, r, k)
                Bz[b_ind] = k

    c = np.concatenate((np.zeros(1), -c))
    A = np.vstack((c.reshape((1, len(c))), A[1:, 0:n+1]))

    for i, s in enumerate(Bz, start=1):
        if A[0, s] != 0:
            assert A[i, s] == 1
            A[0] -= A[i] * A[0, s]

    first_opt_solution = np.zeros(n)
    for s in Bz:
        first_opt_solution[s - 1] = A[s, 0]

    # second phase

if __name__ == "__main__":
    A = np.array([[1, 1, -1, 3],
                  [1, -2, 3, -1],
                  [5, -4, 7, 3]])
    b = np.array([1, 1, 5])
    c = np.array([-1, -1, 0, -5])
    simplex_method(A, b, c)