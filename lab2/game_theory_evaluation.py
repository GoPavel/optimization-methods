import numpy as np

from simplex_method import simplex_method


def print_eval(A, b, c):
    print(f'Sample:\nA:\n{A}\nb:{b}, c:{c}\n')
    res, x = simplex_method(A, b, c, leq=False)
    print(f'Maximum: {res} on {x}')

    print(f'Result: {1 / res}\nx: {x / res} (sum: {np.sum(x / res)})')


def main():
    A = np.array([[0, 1, 2, 3],
                  [1, 0, 1, 2],
                  [2, 1, 0, 1],
                  [3, 2, 1, 0]])
    b = np.array([1, 1, 1, 1])
    c = np.array([1, 1, 1, 1])
    print('>>> First player <<<')
    print_eval(A, b, c)

    A = (-A + 3)
    print('>>> Second player <<<')
    print_eval(A, b, c)


if __name__ == '__main__':
    main()
