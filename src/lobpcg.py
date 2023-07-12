import time

from icecream import ic
import numpy as np
from scipy.linalg import eigh, eigvalsh, qr
from scipy.sparse import diags, random, spdiags
import scipy.sparse.linalg as linalg
from scipy.sparse.linalg import eigs, eigsh, inv


def rand_symm_mat(n=10, eig_low=0.1, eig_high=100.0, nrepeat=1):
    # Randomly generated matrix that will be used to generate the eigenvectors
    QRmat = -1.0 + 2 * np.random.uniform(size=(n, n))

    Q, _ = np.linalg.qr(QRmat, mode="complete")  # Construct Q via a Q-R decomposition

    if nrepeat == 1:
        lam = np.random.uniform(low=eig_low, high=eig_high, size=n)
    else:
        lam = np.hstack(
            (
                eig_low * np.ones(nrepeat),
                np.random.uniform(low=eig_low, high=eig_high, size=n - nrepeat),
            )
        )

    return np.dot(Q, np.dot(np.diag(lam), Q.T))  # Compute A = Q*Lambda*Q^{T}


def lobpcg(A, X, B=None, M=None, tol=1e-8, maxiter=500):
    N, m = X.shape

    if B is None:
        B = np.eye(N)

    # Normalize X.
    # X = qr(X, mode="economic")[0]
    P = np.zeros((N, m))
    tol = 1e-8
    k = 0
    residual = 1

    XBX = np.zeros(m)
    XAX = np.zeros(m)
    mu = np.zeros(m)

    # ic(A)
    # ic(X)
    # ic(B)
    while k < maxiter and residual > tol:
        AX = np.dot(A, X)
        BX = np.dot(B, X)
        # ic(AX)
        # ic(BX)
        # compute the inner product for each column of X
        for i in range(m):
            XBX[i] = np.dot(X[:, i].T, BX[:, i])
            # ic(XBX[i])
            XAX[i] = np.dot(X[:, i].T, AX[:, i])
            # ic(XAX[i])
            mu[i] = XBX[i] / XAX[i]
            # ic(mu[i])

        # ic(mu)
        # Compute residuals.
        R = BX - AX * mu
        # ic(R)
        TR = M @ R
        # ic(TR)
        # W = qr(TR, mode="economic")[0]
        W = TR
        # ic(W)
        
        

        # Perform Rayleigh-Ritz procedure.

        # Compute symmetric Gram matrices.
        if k > 0:
            # P = qr(P, mode="economic")[0]
            Z = np.hstack((W, X, P))
        else:
            Z = np.hstack((W, X))
            
        # ic(Z)

        gramA = np.dot(Z.T, np.dot(A, Z))
        gramB = np.dot(Z.T, np.dot(B, Z))
        
        # ic(gramA)
        # ic(gramB) 

        # Solve generalized eigenvalue problem
        # lambda_, Y = eigsh(gramA, k=m, M=gramB, sigma=1, which="LM")
        lambda_, Y = eigh(gramA, gramB)
        # sort eigenvalues and eigenvectors
        loc = np.argsort(lambda_)
        lambda_ = lambda_[loc[0:m]]
        Y = Y[:, loc[0:m]]
        
        # ic(lambda_)
        # ic(Y)
        # Compute Ritz vectors.
        Yw = Y[:m, :]
        
        # ic(Yw)
        
        Yx = Y[m : 2 * m, :]
        
        # ic(Yx)
        if k > 0:
            Yp = Y[2 * m :, :]
        else:
            Yp = np.zeros((m, m))
            
        # ic(Yp)

        # X = np.dot(W, Yw) + np.dot(X, Yx) + np.dot(P, Yp)
        P = np.dot(W, Yw) + np.dot(P, Yp)
        # ic(X)
        # ic(np.dot(X, Yx))
        X = P + np.dot(X, Yx)
        
        # ic(X)
        # ic(P)

        residual = np.linalg.norm(R)
        ic(k, residual)

        k += 1

    eigenval = lambda_
    eigenvec = X
    return eigenval, eigenvec


if __name__ == "__main__":
    n = 10000  # Size of the matrix
    m = 10  # Number of desired eigenpairs
    np.random.seed(0)

    # A = rand_symm_mat(n=n, eig_low=2, eig_high=10, nrepeat=1)
    # B = rand_symm_mat(n=n, eig_low=1, eig_high=2, nrepeat=1)
    A = np.random.rand(n, n)
    A = A + A.T
    A = A + n * np.eye(n)
    
    B = np.random.rand(n, n)
    B = B + B.T
    B = B + n * np.eye(n)
    
    # A = diags([1, 10, 1], [-1, 0, 1], shape=(n, n)).toarray()
    # A = diags(np.arange(2, n + 2), 0, shape=(n, n)).toarray()
    # B = np.eye(n)
    
    M = np.linalg.inv(A)
    # X = np.random.rand(n, m)
    X = np.eye(n, m)
    
    # Call the lobpcg function
    start = time.time()
    lam1, vec1 = lobpcg(A, X, B, M=M)
    end = time.time()
    t1 = end - start

    start = time.time()
    lam2, vec2 = eigh(A, B)
    end = time.time()
    t2 = end - start
    loc2 = np.argsort(lam2)[0]

    start = time.time()
    lam3, vec3 = linalg.lobpcg(A, X, B=B, M=M, largest=False, tol=1e-8, maxiter=500)
    end = time.time()
    t3 = end - start

    start = time.time()
    lam4, vec4 = eigsh(A, k=m, M=B, sigma=0.1, which="LM")
    end = time.time()
    t4 = end - start

    # Print the resulting eigenvalues and eigenvectors
    print()
    print("Eigenvalues (my::lobpcg):      ", lam1)
    print("Eigenvalues (scipy::eigh):     ", np.sort(lam2)[:m])
    print("Eigenvalues (scipy::lobpcg):   ", lam3)
    print("Eigenvalues (scipy::eigsh):    ", lam4)
    print()
    print("Eigenvectors (my::lobpcg):     ", vec1[:m, 0])
    print("Eigenvectors (scipy::eigh):    ", vec2[:m, loc2])
    print("Eigenvectors (scipy::lobpcg):  ", vec3[:m, 0])
    print("Eigenvectors (scipy::eigsh):   ", vec4[:m, 0])
    print()
    print("Time elapsed (my::lobpcg):      {}".format(t1))
    print("Time elapsed (scipy::eigh):     {}".format(t2))
    print("Time elapsed (scipy::lobpcg):   {}".format(t3))
    print("Time elapsed (scipy::eigsh):    {}".format(t4))
