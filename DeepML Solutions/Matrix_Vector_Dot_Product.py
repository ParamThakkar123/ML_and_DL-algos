def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:

	if not a or not a[0]:
		return -1

	m = len(a)
	n = len(a[0])
	o = len(b)

	if o != n:
		return -1

	res = [0] * n

	for i in range(n):
		res[i] = sum([a[i][j] * b[j] for j in range(n)])

	return res

# Numpy Version

import numpy as np
def matrix_dot_vector_numpy(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:

	if not a or not a[0]:
		return -1

	a_np = np.array(a)
	b_np = np.array(b)

	if a_np.shape[1] != b_np.shape[0]:
		return -1

	return a_np @ b_np