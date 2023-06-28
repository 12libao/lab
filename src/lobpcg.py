import time

from icecream import ic
import numpy as np
from scipy.linalg import eigh, eigvalsh, qr
from scipy.sparse import diags, random
from scipy.sparse.linalg import eigsh, inv


def lobpcg_impl(X, A, M):
    N, m = X.shape  # N = 1000, m = 10

    maxiters = 100
    # Normalize X.
    X, _ = qr(X, mode="economic")
    P = np.zeros((N, m))

    for k in range(maxiters):
        numerator = np.zeros(m)
        # compute the inner product for each column of X
        for i in range(m):
            numerator[i] = X[:, i].T @ M @ X[:, i]
      
        mu = np.dot(X.T, X) / np.dot(X.T, np.dot(A, X))
        ic(mu.shape)
        R = X - mu * np.dot(A, X)
        W, _ = qr(R, mode="economic")

        # Perform Rayleigh-Ritz procedure.

        # Compute symmetric Gram matrices.
        if k > 0:
            P, _ = qr(P, mode="economic")
            Z = np.hstack((W, X, P))
        else:
            Z = np.hstack((W, X))

        gramA = np.dot(Z.T, np.dot(A, Z))
        gramB = np.dot(Z.T, Z)

        # Solve generalized eigenvalue problem.
        lambda_, Y = eigh(gramA, gramB, subset_by_index=[0, m - 1])

        # Compute Ritz vectors.
        Yw = Y[:m, :]
        Yx = Y[m : 2 * m, :]
        if k > 0:
            Yp = Y[2 * m :, :]
        else:
            Yp = np.zeros((m, m))
        X = np.dot(W, Yw) + np.dot(X, Yx) + np.dot(P, Yp)
        P = np.dot(W, Yw) + np.dot(P, Yp)

    eigenval = lambda_
    eigenvec = X
    return eigenvec, eigenval


if __name__ == "__main__":
    # N = 1000
    # A = random(N, N, density=0.1, format='csr')
    # M = random(N, N, density=0.1, format='csr')
    # X = random(N, 10, density=0.1, format='csr')

    # start = time.time()
    # lobpcg_impl(X, A, M)
    # end = time.time()
    # print('Time elapsed: {}'.format(end - start))

    # start = time.time()
    # eigsh(A, 10, M, which='SA')
    # end = time.time()
    # print('Time elapsed: {}'.format(end - start))

    # Generate some sample data
    N = 100  # Size of the matrix
    m = 5  # Number of desired eigenvalues/vectors
    X = np.random.rand(N, m)  # Initial guess for eigenvectors
    A = diags(
        [1, -2, 1], [-1, 0, 1], shape=(N, N)
    ).toarray()  # Example symmetric matrix
    M = inv(
        diags([1, 2, 1], [-1, 0, 1], shape=(N, N))
    )  # Example preconditioning matrix

    # Call the lobpcg_impl function
    eigenvec, eigenval = lobpcg_impl(X, A, M)

    # Print the resulting eigenvalues and eigenvectors
    print("Eigenvalues:")
    print(eigenval)
    print("\nEigenvectors:")
    print(eigenvec)

    # Alternatively, you can compare the result with SciPy's eigvalsh function
    scipy_eigenval = eigvalsh(A, M)
    print("\nEigenvalues (using eigvalsh):")
    print(scipy_eigenval)
