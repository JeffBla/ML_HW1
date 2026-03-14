import copy
import math
from typing import NamedTuple


class Point(NamedTuple):
    x: float
    y: float


class Matrix:

    def __init__(self, n, m):
        self.n = n  # row
        self.m = m  # col
        self.mat = self.createMatArray(n, m)

    def createMatArray(self, n, m):
        return [0 for _ in range(n * m)]

    def identity(self):
        if self.n == self.m:
            for i in range(self.n):
                self[i, i] = 1
        else:
            raise ValueError("Identity not doable")

    def transpose(self):
        new_M = Matrix(self.m, self.n)

        for i in range(self.n):
            for j in range(self.m):
                new_M[j, i] = self[i, j]

        return new_M

    def __mul__(self, other):
        if isinstance(other, Matrix):
            if self.m != other.n:
                raise ValueError(
                    "Matrix dimensions do not match for multiplication")

            new_M = Matrix(self.n, other.m)
            for i in range(self.n):
                for j in range(other.m):
                    v = 0
                    for k in range(self.m):
                        v += self[i, k] * other[k, j]
                    new_M[i, j] = v
        else:  # scalar mul
            new_M = Matrix(self.n, self.m)
            for i in range(self.n):
                for j in range(self.m):
                    new_M[i, j] = self[i, j] * other
        return new_M

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if self.n != other.n or self.m != other.m:
            raise ValueError("Matrix dimensions do not match for addition")

        new_M = Matrix(self.n, self.m)
        for i in range(self.n):
            for j in range(self.m):
                new_M[i, j] = self[i, j] + other[i, j]
        return new_M

    def __sub__(self, other):
        if self.n != other.n or self.m != other.m:
            raise ValueError("Matrix dimensions do not match for substraction")

        new_M = Matrix(self.n, self.m)
        for i in range(self.n):
            for j in range(self.m):
                new_M[i, j] = self[i, j] - other[i, j]
        return new_M

    def __eq__(self, other):
        if self.n != other.n or self.m != other.m:
            return False
        for i in range(len(self.mat)):
            if self[i] != other[i]:
                return False
        return True

    def norm(self):
        return math.sqrt(sum([x**2 for x in self.mat]))

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            i, j = key
            if i < 0 or i >= self.n or j < 0 or j >= self.m:
                raise IndexError("Matrix index out of range")
            self.mat[i * self.m + j] = value
        elif isinstance(key, int):
            if key < 0 or key >= len(self.mat):
                raise IndexError("Matrix index out of range")
            self.mat[key] = value
        else:
            raise TypeError("Invalid index type")

    def __getitem__(self, key):
        if isinstance(key, tuple):
            i, j = key
            if i < 0 or i >= self.n or j < 0 or j >= self.m:
                raise IndexError("Matrix index out of range")
            return self.mat[i * self.m + j]
        elif isinstance(key, int):
            if key < 0 or key >= len(self.mat):
                raise IndexError("Matrix index out of range")
            return self.mat[key]
        else:
            raise TypeError("Invalid index type")

    def inverse(self):
        if self.n != self.m:
            raise ValueError("Only square matrices can be inverted")

        copied_mat = copy.deepcopy(self.mat)
        inversed_M = Matrix(self.n, self.m)
        inversed_M.identity()

        for i in range(self.n):
            # if pivot is 0, swap with a lower row
            if copied_mat[i * self.m + i] == 0:
                swap_row = -1
                for r in range(i + 1, self.n):
                    if copied_mat[r * self.m + i] != 0:
                        swap_row = r
                        break

                if swap_row == -1:
                    raise ValueError("Matrix is singular and not invertible")

                for c in range(self.m):
                    copied_mat[i * self.m + c], copied_mat[swap_row * self.m + c] = \
                        copied_mat[swap_row * self.m + c], copied_mat[i * self.m + c]

                    inversed_M[i, c], inversed_M[swap_row, c] = \
                        inversed_M[swap_row, c], inversed_M[i, c]

            # make mat[i][i] = 1
            divisor = copied_mat[i * self.m + i]
            for j in range(self.m):
                copied_mat[i * self.m + j] /= divisor
                inversed_M[i, j] /= divisor

            # subtract to remove value in the same rank
            for j in range(self.n):
                if i != j:
                    subtractor = copied_mat[j * self.m + i]
                    for k in range(self.m):
                        copied_mat[j * self.m + k] -= \
                            subtractor * copied_mat[i * self.m + k]
                        inversed_M[j, k] -= \
                            subtractor * inversed_M[i, k]

        return inversed_M
