from Matrix import *
from pathlib import Path


def sign(M: Matrix):
    res = Matrix(M.n, M.m)
    for i in range(len(M.mat)):
        if M[i] > 0:
            res[i] = 1
        elif M[i] < 0:
            res[i] = -1
        else:
            res[i] = 0
    return res


def readpts(filepath) -> list:
    data_file = Path(filepath)
    points = []
    with open(data_file, "r") as f:
        for line in f:
            x, y = line.split(',')
            pt = Point(float(x), float(y))
            points.append(pt)
    return points


def makeA(n_coef, points):
    A = Matrix(len(points), n_coef)
    for i in range(A.n):
        x = points[i].x
        for j in range(A.m):
            A.mat[i * A.m + j] = pow(x, j)
    return A


def makeB(points):
    B = Matrix(len(points), 1)
    for i in range(B.n):
        B.mat[i] = points[i].y
    return B


def plotPolynomial(points, coef: Matrix):
    import numpy as np
    from matplotlib import pyplot as plt

    x_min = min(points, key=lambda p: p.x).x
    x_max = max(points, key=lambda p: p.x).x
    x = np.linspace(x_min - 1, x_max + 1, 200)

    y = np.polyval(list(reversed(coef.mat)), x)

    pt_x = [p.x for p in points]
    pt_y = [p.y for p in points]

    # Plot the polynomial using Matplotlib
    plt.figure(figsize=(8, 6))

    plt.scatter(pt_x, pt_y, color='red', edgecolors='black')

    plt.plot(x, y, color='black')
    plt.title('Polynomial Plot')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')

    plt.savefig("result.png", dpi=300)
    plt.close()


def printPolynomial(coef: Matrix):
    terms = []
    for i, c in reversed(list(enumerate(coef.mat))):
        if i == 0:
            terms.append(f"{c}")
        elif i == 1:
            terms.append(f"{c} x")
        else:
            terms.append(f"{c} x^{i}")
    print("Fitting line:", " + ".join(terms))


def totalError(points, coef: Matrix):
    error = 0

    for p in points:
        x = p.x
        y_pred = 0

        for i, c in enumerate(coef.mat):
            y_pred += c * (x**i)

        error += (p.y - y_pred)**2

    return error
