import os
import argparse
from random import random
from Matrix import *
from Utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, help="The number of polynomial bases")
    parser.add_argument("reg_lambda",
                        type=float,
                        help="only for LSE and Steepest descent cases")
    parser.add_argument("lr",
                        type=float,
                        help="learning rate for Steepest descent cases")
    parser.add_argument(
        "filepath",
        type=str,
        help="The path and name of a file which consists of data points")
    arg = parser.parse_args()

    points = readpts(arg.filepath)

    A = makeA(arg.n, points)
    b = makeB(points)

    A_T = A.transpose()
    Ab = A_T * b * 2

    x = Matrix(arg.n, 1)
    # Initial point
    for i in range(len(x.mat)):
        x[i] = random()
    eps = 1e-6
    epoch = 10000
    for i in range(epoch):
        AAx = A_T * A * x * 2

        grad = AAx - Ab + arg.reg_lambda * sign(x)

        new_x = arg.lr * (-1 * grad) + x

        if (new_x - x).norm() < eps:
            break

        x = new_x

    plotPolynomial(points, x)

    printPolynomial(x)
    print(f"Total error: {totalError(points, x)}")
