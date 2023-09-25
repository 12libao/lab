from icecream import ic
import numpy as np
from numpy import array
from scipy.sparse import coo_array
import time


def coo_to_csr(Ax, Ai, Aj):
    # Determine the number of rows and columns
    num_rows = max(Ai) + 1

    # Create a dictionary to store (row, col) pairs as keys and sum Bx as Bx
    coo_dict = {}

    # Sum duplicate entries and compute the number of non-zero elements in each row
    row_counts = [0] * num_rows
    for i in range(len(Ax)):
        row = Ai[i]
        col = Aj[i]
        key = (row, col)
        if key in coo_dict:
            coo_dict[key] += Ax[i]
        else:
            coo_dict[key] = Ax[i]
            row_counts[row] += 1
            
    # Compute the Bp array
    Bp = [0]
    Bp.extend(np.cumsum(row_counts))

    # Fill in the CSR arrays
    Bx = list(coo_dict.values())
    Bj = [key[1] for key in coo_dict.keys()]

    return Bx, Bp, Bj


# Sample COO matrix Ax
Ax = [1, 1, 1, 1, 1, 1, 1]
row = [0, 0, 1, 3, 1, 0, 0]
col = [0, 2, 1, 3, 1, 0, 0]

A = coo_array((Ax, (row, col)), shape=(4, 4)).tocsr()
ic(A.data, A.indices, A.indptr)
ic(A.toarray())

# Convert COO to CSR format
csr_values, csr_indices, csr_indptr = coo_to_csr(Ax, row, col)

# Print the CSR format arrays
print("CSR Values:", csr_values)
print("CSR Indices:", csr_indices)
print("CSR Indptr:", csr_indptr)

# generate a random sparse matrix
n = 1000000
nnz = 10000000

Ax = np.random.rand(nnz)
Ai = np.random.randint(0, n, nnz)
Aj = np.random.randint(0, n, nnz)

# convert to CSR format
t0 = time.time()
Bx, Bp, Bj = coo_to_csr(Ax, Ai, Aj)
t1 = time.time()
print("coo_to_csr:", t1 - t0)

# convert to CSR format using scipy
t0 = time.time()
C = coo_array((Ax, (Ai, Aj)), shape=(n, n)).tocsr()
t1 = time.time()
print("scipy coo_to_csr:", t1 - t0)
