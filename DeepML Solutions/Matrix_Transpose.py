def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    if not a:
        return []

    rows = len(a)
    cols = len(a[0])

    res = [[0 for _ in range(rows)] for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            res[j][i] = a[i][j]
    return res

# Numpy version
import numpy as np
def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    return np.array(a).T.tolist()