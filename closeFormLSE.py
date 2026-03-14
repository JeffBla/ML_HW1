import os
import argparse
from Matrix import *
from Utils import *


def closeFormLSE(n_coef, points):
    A = makeA(n_coef, points)
    A_T = A.transpose()
    AA = A_T * A
    I = Matrix(AA.n, AA.m)
    I.identity()
    AAI = AA + I * arg.reg_lambda
    inversed_AAI = AAI.inverse()
    AAAI = inversed_AAI * A_T
    b = makeB(points)
    x = AAAI * b
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

    x = closeFormLSE(arg.n, points)

    plotPolynomial(points, x)

    printPolynomial(x)
    print(f"Total error: {totalError(points, x)}")
