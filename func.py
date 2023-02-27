import math
import numpy as np
def corcoeff(xd, yd):
    sigma1 = sigma_xy(xd, yd) * len(xd)
    sigma2 = sum(xd) * sum(yd)
    sigma3 = len(xd) * sum([val ** 2 for val in xd])
    sigma4 = sum(xd) ** 2
    sigma5 = len(yd) * sum([val ** 2 for val in yd])
    sigma6 = sum(yd) ** 2
    top = (sigma1 - sigma2)
    bottom = (math.sqrt(sigma3 - sigma4)) * (math.sqrt(sigma5 - sigma6))
    return top / bottom


def sigma_xy(xd, yd):
    nlist = []
    for i in range(len(xd)):
        nlist.append((xd[i] * yd[i]))
    return sum(nlist)


def least_sqrs(xd, yd):
    matrix1 = [[sum(val ** 2 for val in xd), sum(xd)], [sum(xd), len(xd)]]
    matrix2 = [sigma_xy(xd, yd), sum(yd)]
    array1 = np.array(matrix1)
    array2 = np.array(matrix2)
    invarray1 = np.linalg.inv(array1)
    solution = np.dot(invarray1, array2)
    return solution