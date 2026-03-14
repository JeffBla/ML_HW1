import os
import argparse
from random import random
from Matrix import *
from Utils import *


def newtonLSE(n_coef, points):
    A = makeA(n_coef, points)
    b = makeB(points)

    A_T = A.transpose()
    Ab = A_T * b * 2
    # hessian
    h = A_T * A * 2

    x = Matrix(arg.n, 1)
    # Initial point
    for i in range(len(x.mat)):
        x[i] = random()
    eps = 1e-6
    epoch = 100
    for i in range(epoch):
        # grad
        AAx = A_T * A * x * 2

        grad = AAx - Ab

        new_x = h.inverse() * grad * (-1) + x

        if (new_x - x).norm() < eps:
            break

        x = new_x
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, help="The number of polynomial bases")
    parser.add_argument("reg_lambda",
                        type=int,
                        help="only for LSE and Steepest descent cases")
    parser.add_argument(
        "filepath",
        type=str,
        help="The path and name of a file which consists of data points")
    arg = parser.parse_args()

    points = readpts(arg.filepath)

    x = newtonLSE(arg.n, points)

    plotPolynomial(points, x)

    printPolynomial(x)
    print(f"Total error: {totalError(points, x)}")
