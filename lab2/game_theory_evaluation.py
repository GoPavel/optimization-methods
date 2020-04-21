import numpy as np

from simplex_method import simplex_method


def main():
    A = np.array([[0, 1, 2, 3],
                  [1, 0, 1, 2],
                  [2, 1, 0, 1],
                  [3, 2, 1, 0]])
    b = [1, 1, 1, 1]
    c = [1, 1, 1, 1]
    print(f'Sample:\nA:\n{A}\nb:{b}, c:{c}\n')
    res, x = simplex_method(A, b, c, leq=False)
    print(f'Minimum: {res} on {x}')

    print(f'Result: {1 / res}\nx: {x / res}\nsum: {np.sum(x/res)}')

if __name__ == '__main__':
    main()
