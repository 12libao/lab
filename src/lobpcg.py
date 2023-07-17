import time

from icecream import ic
import numpy as np
from scipy.linalg import eigh, eigvalsh, qr, solve_triangular
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


def RayleighRitz(A, B, X, m):
    XAX = X.T @ A @ X
    XBX = X.T @ B @ X
    eigval, eigvec = eigh(XAX, XBX, subset_by_index=[0, m - 1])
    return eigvec, np.diag(eigval)


def RayleighRitz2(A, B, X, m):
    XAX = X.T @ A @ X
    XBX = X.T @ B @ X
    D = np.diag(1 / np.sqrt(np.diag(XBX)))

    L = np.linalg.cholesky(D @ XBX @ D)
    L = np.linalg.inv(L)
    R = D @ L.T
    A_new = R.T @ XAX @ R
    eigval, eigvec = eigh(A_new, subset_by_index=[0, m - 1])
    C = R @ eigvec
    return C, np.diag(eigval)


def RayleighRitz3(XAX, XBX, m):
    d = 1 / np.sqrt(np.diag(XBX))
    d_row = d.reshape(-1, 1)
    # d_col = d.reshape(1, -1)
    D = np.outer(d, d)

    DXBX = XBX * D

    # Cholesky decomposition of DXBX
    L = np.linalg.cholesky(DXBX)
    L = np.linalg.inv(L)
    R = L.T * d_row
    A_new = R.T @ XAX @ R
    eigval, eigvec = eigh(A_new, subset_by_index=[0, m - 1])
    C = R @ eigvec
    return C, np.diag(eigval)


def RayleighRitz4(XBX):
    d = 1 / np.sqrt(np.diag(XBX))
    d_row = d.reshape(-1, 1)
    # d_col = d.reshape(1, -1)
    D = np.outer(d, d)

    DXBX = XBX * D

    # Cholesky decomposition of DXBX
    L = np.linalg.cholesky(DXBX)
    # L = np.linalg.inv(L)
    return solve_triangular(L.T, d)


def svqb(M, U, tol_replace=1e-8):
    """
    Compute the SVD of U with respect to M.

    Parameters
    ----------
        M : symmetric positive definite matrix, size is (n, n)
        U : candidate basis, size is (n, m)

    Returns
    -------
        U : M orthonormal columns (:math:`U^T M U = I`)
    """
    UMU = U.T @ M @ U
    D = np.diag(1 / np.sqrt(np.diag(UMU)))

    DUMUD = D @ UMU @ D
    O, Z = eigh(DUMUD)

    O = np.maximum(O, tol_replace * np.max(np.abs(O)))

    O = np.diag(1 / np.sqrt(O))

    return U @ D @ Z @ O


def svqbDrop(M, U, tol_drop=1e-8):
    """
    Compute the SVD of U with respect to M.

    Parameters
    ----------
        M : symmetric positive definite matrix, size is (n, n)
        U : candidate basis, size is (n, m)

    Returns
    -------
        U : M orthonormal columns (:math:`U^T M U = I`)
    """
    UMU = U.T @ M @ U
    D = np.diag(1 / np.sqrt(np.diag(UMU)))

    DUMUD = D @ UMU @ D
    O, Z = eigh(DUMUD)

    # Determine columns to keep J={ j:O j> tol_drop * maxi(|O| )}
    J = np.where(O > tol_drop * np.max(np.abs(O)))[0]
    Z = Z[:, J]
    O = np.diag(1 / np.sqrt(O[J]))

    return U @ D @ Z @ O


def ortho(M, U, V, itmax1=3, itmax2=3, tol_replace=1e-8, tol_ortho=1e-8):
    """
    Orthogonalize U against V with respect to M.
        U.T @ M @ V = 0 and U.T @ M @ U = I

    Parameters
    ----------
        M : symmetric positive definite matrix, size is (n, n)
        U : candidate basis, size is (n, m)
        V : M-orthogonal external basis, size is (n, k)

    Returns
    -------
        U : M-orthonormal columns (:math:`U^T M U = I`) such that :math:`V^T M U=0`
    """

    MV_norm = np.linalg.norm(M @ V)
    VMU = V.T @ M @ U

    rerr = 1
    i = j = 0
    for i in range(itmax1):
        U = U - V @ (VMU)

        for j in range(itmax2):
            U = svqb(M, U, tol_replace)
            MU = M @ U
            UMU = U.T @ MU
            U_norm = np.linalg.norm(U)
            MU_norm = np.linalg.norm(M @ U)
            R = UMU - np.eye(UMU.shape[0])
            R_norm = np.linalg.norm(R)
            rerr = R_norm / (U_norm * MU_norm)
            if rerr < tol_replace:
                break

        VMU = V.T @ MU
        VMU_norm = np.linalg.norm(VMU)
        rerr = VMU_norm / (MV_norm * U_norm)
        if rerr < tol_ortho:
            break

    return U


def orthoDrop(
    M, U, V, itmax1=3, itmax2=3, tol_replace=1e-8, tol_drop=1e-8, tol_ortho=1e-8
):
    """
    Orthogonalize U against V with respect to M.
        U.T @ M @ V = 0 and U.T @ M @ U = I

    Parameters
    ----------
        M : symmetric positive definite matrix, size is (n, n)
        U : candidate basis, size is (n, m)
        V : M-orthogonal external basis, size is (n, k)

    Returns
    -------
        U : M-orthonormal columns (:math:`U^T M U = I`) such that :math:`V^T M U=0`
    """

    MV_norm = np.linalg.norm(M @ V)
    VMU = V.T @ M @ U

    rerr = 1
    i = j = 0
    for i in range(itmax1):
        U = U - V @ (VMU)

        for j in range(itmax2):
            if j == 0:
                U = svqb(M, U, tol_replace)
            else:
                U = svqbDrop(M, U, tol_drop)

            MU = M @ U
            UMU = U.T @ MU
            U_norm = np.linalg.norm(U)
            MU_norm = np.linalg.norm(M @ U)
            R = UMU - np.eye(U.shape[1])
            R_norm = np.linalg.norm(R)
            rerr = R_norm / (U_norm * MU_norm)
            if rerr < tol_replace:
                break

        VMU = V.T @ MU
        VMU_norm = np.linalg.norm(VMU)
        rerr = VMU_norm / (MV_norm * U_norm)
        if rerr < tol_ortho:
            break

    return U


def lobpcg(A, X, B=None, M=None, tol=1e-8, maxiter=500):
    N, m = X.shape

    if B is None:
        B = np.eye(N)

    # Normalize X.
    # X = X / np.linalg.norm(X)
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
        # X = X / np.linalg.norm(X)
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
        W = R
        # W = W / np.linalg.norm(W)

        # ic(W)

        # Perform Rayleigh-Ritz procedure.

        # Compute symmetric Gram matrices.
        if k > 0:
            # P = P / np.linalg.norm(P)
            Z = np.hstack((X, W, P))
        else:
            Z = np.hstack((X, W))

        # ic(Z)
        # z_norm = np.linalg.norm(Z)
        # z_norm = 1
        # Z = Z

        gramA = np.dot(Z.T, np.dot(A, Z))
        gramB = np.dot(Z.T, np.dot(B, Z))

        # ic(gramA)
        # ic(gramB)

        # Solve generalized eigenvalue problem
        # lambda_, Y = eigsh(gramA, k=m, M=gramB, sigma=1, which="LM")
        lambda_, Y = eigh(gramA, gramB, subset_by_index=[0, m - 1])

        # ic(lambda_)
        # ic(Y)
        # Compute Ritz vectors.
        Yx = Y[:m, :]

        # ic(Yw)

        Yw = Y[m : 2 * m, :]

        # ic(Yx)
        if k > 0:
            Yp = Y[2 * m :, :]
        else:
            Yp = np.zeros((m, m))

        # ic(Yp)

        # X = np.dot(W, Yw) + np.dot(X, Yx) + np.dot(P, Yp)
        # P = np.dot(W, Yw) + np.dot(P, Yp)
        P = Z[:, m:] @ Y[m:, :]
        # ic(X)
        # ic(np.dot(X, Yx))
        X = Z @ Y

        # ic(X)
        # ic(P)

        residual = np.linalg.norm(R)
        ic(k, residual)

        k += 1

    eigenval = lambda_
    eigenvec = X
    return eigenval, eigenvec


def lobpcg2(A, X, B=None, M=None, tol=1e-8, maxiter=500):
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

    while k < maxiter and residual > tol:
        AX = np.dot(A, X)
        BX = np.dot(B, X)
        # compute the inner product for each column of X
        for i in range(m):
            XBX[i] = np.dot(X[:, i].T, BX[:, i])
            # ic(XBX[i])
            XAX[i] = np.dot(X[:, i].T, AX[:, i])
            # ic(XAX[i])
            mu[i] = XBX[i] / XAX[i]
            # ic(mu[i])

        R = BX - AX * mu
        W = R

        # Perform Rayleigh-Ritz procedure.

        # Compute symmetric Gram matrices.
        X_hat = np.zeros((N, m))
        for i in range(m):
            if k > 0:
                z = np.vstack((W[:, i], X[:, i], P[:, i])).T
            else:
                z = np.vstack((W[:, i], X[:, i])).T

            gramA = np.dot(z.T, np.dot(A, z))
            gramB = np.dot(z.T, np.dot(B, z))

            _, Y = eigh(gramA, gramB, subset_by_index=[0, 0])

            if k > 0:
                yp = Y[2]
            else:
                yp = 0

            X_hat[:, i] = W[:, i] + Y[1] * X[:, i] + yp * P[:, i]
            P[:, i] = W[:, i] + yp * P[:, i]

        gramA = np.dot(X_hat.T, np.dot(A, X_hat))
        gramB = np.dot(X_hat.T, np.dot(B, X_hat))
        lambda_, Y = eigh(gramA, gramB)

        X = X_hat @ Y

        residual = np.linalg.norm(R)
        ic(k, residual)

        k += 1

    eigenval = lambda_
    eigenvec = X
    return eigenval, eigenvec


def lobpcg3(A, X, B=None, M=None, tol=1e-8, maxiter=500):
    N, m = X.shape

    if B is None:
        B = np.eye(N)

    # A_00 = A[:m, :m]
    # B_00 = B[:m, :m]
    # X_00 = X[:m, :m]
    # ic(X_00.shape)
    # ic(X.shape)
    # X_00, O = RayleighRitz2(A_00, B_00, X_00, m)
    # ic(X_00.shape)
    # X = np.vstack((X_00, X[m:, :]))

    # X = X / np.linalg.norm(X)
    # XAX = X.T @ A @ X
    # XBX = X.T @ B @ X
    # C, O = RayleighRitz3(XAX, XBX, m)

    XAX = A[:m, :m]
    XBX = B[:m, :m]
    # C, O = RayleighRitz3(XAX, XBX, m)
    O, C = eigh(XAX, XBX)
    O = np.diag(O)

    X = X @ C

    R = A @ X - B @ X @ O
    P = np.zeros((N, m))

    tol = 1e-8
    k = 0
    residual = 1
    # M = np.linalg.inv(A)

    while k < maxiter and residual > tol:
        # W = M @ R
        W = R
        # if k > 0:
        #     W = ortho(B, R, np.hstack((X, P)))
        # else:
        #     W = ortho(B, R, X)

        # W = W / np.linalg.norm(W)

        if k > 0:
            # P = P / np.linalg.norm(P)
            S = np.hstack((X, W, P))
            # AP = A @ P
            # BP = B @ P
        else:
            S = np.hstack((X, W))

        # AX = A @ X
        # AW = A @ W
        # BX = B @ X
        # BW = B @ W

        # if k > 0:
        #     XAX = O
        #     XAW = X.T @ AW
        #     XAP = X.T @ AP
        #     WAW = W.T @ AW
        #     WAP = W.T @ AP
        #     PAP = P.T @ AP

        #     XBX = np.eye(m)
        #     XBW = X.T @ BW
        #     XBP = X.T @ BP
        #     WBW = W.T @ BW
        #     WBP = W.T @ BP
        #     PBP = P.T @ BP

        #     SAS = np.vstack(
        #         (
        #             np.hstack((XAX, XAW, XAP)),
        #             np.hstack((XAW.T, WAW, WAP)),
        #             np.hstack((XAP.T, WAP.T, PAP)),
        #         )
        #     )
        #     SBS = np.vstack(
        #         (
        #             np.hstack((XBX, XBW, XBP)),
        #             np.hstack((XBW.T, WBW, WBP)),
        #             np.hstack((XBP.T, WBP.T, PBP)),
        #         )
        #     )
        # else:
        #     XAX = O
        #     XAW = X.T @ AW
        #     WAW = W.T @ AW

        #     XBX = np.eye(m)
        #     XBW = X.T @ BW
        #     WBW = W.T @ BW

        #     SAS = np.vstack((np.hstack((XAX, XAW)), np.hstack((XAW.T, WAW))))
        #     SBS = np.vstack((np.hstack((XBX, XBW)), np.hstack((XBW.T, WBW))))

        # S = svqb(B, S)

        # ic(S.T @ B @ S)
        # ic(X.T @ A @ P)
        # ic(X.T @ B @ P)

        # ic(P.T @ B @ P)
        # ic(W.T @ B @ W)

        SAS = S.T @ A @ S
        SBS = S.T @ B @ S

        O, C = eigh(SAS, SBS, subset_by_index=[0, m - 1])
        O = np.diag(O)

        # SAS = S.T @ A @ S
        # SBS = S.T @ B @ S

        # C, O = RayleighRitz3(SAS, SBS, m)

        X = S @ C
        R = A @ X - B @ X @ O
        P = S[:, m:] @ C[m:, :]

        residual = np.linalg.norm(R)
        k += 1

        ic(k, residual)

    return O.diagonal(), X


def lobpcg4(A, X, B=None, M=None, tol=1e-8, maxiter=500):
    N, m0 = X.shape
    m00 = m0
    # Basis blocks are padded by about 10\% of the number of desired eigenpairs
    m1 = int(np.ceil(m0 * 0.3))
    
    m = m0 + m1
    ic(m0, m1)

    if B is None:
        B = np.eye(N)
    # X = np.eye(N, m) / np.sqrt(N)
    X = np.eye(N, m)
    XAX = A[:m, :m]
    XBX = B[:m, :m]
    # C, O = RayleighRitz3(XAX, XBX, m)
    O, C = eigh(XAX, XBX)
    O = np.diag(O)

    X = X @ C
    X = X / np.linalg.norm(X)
    R = A @ X - B @ X @ O
    P = np.zeros((N, m))

    omega = np.random.randn(N, m)
    Anorm = np.linalg.norm(A @ X) / np.linalg.norm(X)
    Bnorm = np.linalg.norm(B @ X) / np.linalg.norm(X)
    # ic(Anorm)
    # Anorm = np.linalg.norm(A @ X) / np.linalg.norm(X)
    # ic(Anorm)
    # ic(np.linalg.norm(A))

    tol = 1e-8
    k = 0
    residual = 1
    counter = 0

    X_hat = np.zeros((N, m))
    # XAX = np.zeros((m, m))
    # XBX = np.zeros((m, m))
    SAS = np.zeros((3, 3))
    SBS = np.zeros((3, 3))
    eigenvalues = np.zeros(m)
    res_m0 = m0
    non_convergence_indx = np.arange(m)
    # ic(non_convergence_indx)

    for k in range(maxiter):
        W = R
        # if k > 0:
        #     W = ortho(B, R, np.hstack((X, P)))
        # else:
        #     W = ortho(B, R, X)

        # W = W / np.linalg.norm(W)

        # use hard lock technique to lock the converged eigenpairs
        # counter: only update the un-converged eigenpairs
        # for all index in array non_convergence_indx:
        for i in non_convergence_indx:
            # for i in range(counter, m):
            # for i in range(m):
            # ic(i)
            # if k > 0:
            #     # S = np.vstack((X[:, i], W[:, i], P[:, i])).T
            #     AP = A @ P[:, i]
            #     BP = B @ P[:, i]
            # else:
            #     # S = np.vstack((X[:, i], W[:, i])).T
            #     AX = A @ X[:, i]
            #     BX = B @ X[:, i]

            # # S = S / np.linalg.norm(S)
            # # C, O = RayleighRitz2(A, B, S, 1)
            # # AX = A @ X[:, i]
            # AW = A @ W[:, i]
            # BW = B @ W[:, i]

            # if k > 0:
            #     # set the valuse of O into XAX
            #     # ic(O.shape)
            #     # ic(O)
            #     # ic(np.diag(O)[0].item())
            #     XAX = np.diag(O)[0].item()
            #     XAW = X[:, i].T @ AW
            #     XAP = X[:, i].T @ AP
            #     WAW = W[:, i].T @ AW
            #     WAP = W[:, i].T @ AP
            #     PAP = P[:, i].T @ AP

            #     XBX = 1
            #     XBW = X[:, i].T @ BW
            #     XBP = X[:, i].T @ BP
            #     WBW = W[:, i].T @ BW
            #     WBP = W[:, i].T @ BP
            #     PBP = P[:, i].T @ BP

            #     SAS[0, 0] = XAX
            #     SAS[0, 1] = XAW
            #     SAS[0, 2] = XAP
            #     SAS[1, 0] = XAW
            #     SAS[1, 1] = WAW
            #     SAS[1, 2] = WAP
            #     SAS[2, 0] = XAP
            #     SAS[2, 1] = WAP
            #     SAS[2, 2] = PAP

            #     SBS[0, 0] = XBX
            #     SBS[0, 1] = XBW
            #     SBS[0, 2] = XBP
            #     SBS[1, 0] = XBW
            #     SBS[1, 1] = WBW
            #     SBS[1, 2] = WBP
            #     SBS[2, 0] = XBP
            #     SBS[2, 1] = WBP
            #     SBS[2, 2] = PBP

            # else:
            #     XAX = X[:, i].T @ AX
            #     XAW = X[:, i].T @ AW
            #     WAW = W[:, i].T @ AW

            #     XBX = X[:, i].T @ BX
            #     XBW = X[:, i].T @ BW
            #     WBW = W[:, i].T @ BW

            #     SAS[0, 0] = XAX
            #     SAS[0, 1] = XAW
            #     SAS[1, 0] = XAW
            #     SAS[1, 1] = WAW

            #     SBS[0, 0] = XBX
            #     SBS[0, 1] = XBW
            #     SBS[1, 0] = XBW
            #     SBS[1, 1] = WBW

            # ic(SAS)
            # ic(SBS)
            if k > 0:
                S = np.vstack((X[:, i], W[:, i], P[:, i])).T
            else:
                S = np.vstack((X[:, i], W[:, i])).T

            SAS = S.T @ A @ S
            SBS = S.T @ B @ S

            O, C = eigh(SAS, SBS, subset_by_index=[0, 0])

            if k > 0:
                p = C[1, 0] * W[:, i] + C[2, 0] * P[:, i]
            else:
                p = C[1, 0] * W[:, i]

            x = p + C[0, 0] * X[:, i]

            X[:, i] = x
            P[:, i] = p

        # XAX = X[:, non_convergence_indx].T @ A @ X[:, non_convergence_indx]
        # XBX = X[:, non_convergence_indx].T @ B @ X[:, non_convergence_indx]
        # O, C = eigh(XAX, XBX)
        # eigenvalues[non_convergence_indx] = O
        # X[:, non_convergence_indx] = X[:, non_convergence_indx] @ C
        # R[:, non_convergence_indx] = A @ X[:, non_convergence_indx] - B @ X[
        #     :, non_convergence_indx
        # ] @ np.diag(O)

        # XAX = X[:, counter:m].T @ A @ X[:, counter:m]
        # XBX = X[:, counter:m].T @ B @ X[:, counter:m]
        # O, C = eigh(XAX, XBX)
        # eigenvalues[counter:] = O
        # X[:, counter:m] = X[:, counter:m] @ C
        # R[:, counter:m] = A @ X[:, counter:m] - B @ X[:, counter:m] @ np.diag(O)

        XAX = X.T @ A @ X
        XBX = X.T @ B @ X
        O, C = eigh(XAX, XBX)
        eigenvalues = O
        X = X @ C
        R = A @ X - B @ X @ np.diag(O)

        res_max = 0
        non_convergence_indx = []
        for i in range(m0):
            residual = np.linalg.norm(R[:, i]) / (
                (Anorm + Bnorm * eigenvalues[i]) * np.linalg.norm(X[:, i])
            )
            if residual > res_max and i < m0:
                res_max = residual
            if residual > tol and i < m0:
                non_convergence_indx.append(i)

        counter = m0 - len(non_convergence_indx)
        ic(k, counter, res_max)

        if res_max < tol:
            break

        for i in range(m0, m):
            residual = np.linalg.norm(R[:, i]) / (
                (Anorm + Bnorm * eigenvalues[i]) * np.linalg.norm(X[:, i])
            )
            if residual > tol:
                non_convergence_indx.append(i)

    return eigenvalues[:m0], X[:, :m0]


if __name__ == "__main__":
    n = 100  # Size of the matrix
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
    # start = time.time()
    # M = np.linalg.inv(A)
    # end = time.time()
    # t = end - start
    # ic(t)
    # M = A - 500 * np.eye(n)
    M = None
    X = np.random.rand(n, m)
    # X = np.eye(n, m)

    # Call the lobpcg function
    start = time.time()
    lam1, vec1 = lobpcg4(A, X, B, M=M)
    end = time.time()
    t1 = end - start
    ic("my::lobpcg", t1)

    start = time.time()
    lam2, vec2 = eigh(A, B)
    end = time.time()
    t2 = end - start

    start = time.time()
    lam3, vec3 = linalg.lobpcg(A, X, B=B, M=M, largest=False, tol=1e-8, maxiter=500)
    end = time.time()
    t3 = end - start

    # start = time.time()
    # lam4, vec4 = eigsh(A, k=m, M=B, sigma=0.1, which="LM")
    # end = time.time()
    # t4 = end - start

    # Print the resulting eigenvalues and eigenvectors
    print()
    print("Eigenvalues (my::lobpcg):      ", lam1)
    print("Eigenvalues (scipy::eigh):     ", lam2[:m])
    print("Eigenvalues (scipy::lobpcg):   ", lam3)
    # print("Eigenvalues (scipy::eigsh):    ", lam4)
    print()
    print("Eigenvectors (my::lobpcg):     ", vec1[:m, m - 1])
    print("Eigenvectors (scipy::eigh):    ", vec2[:m, m - 1])
    print("Eigenvectors (scipy::lobpcg):  ", vec3[:m, m - 1])
    # print("Eigenvectors (scipy::eigsh):   ", vec4[:m, 0])
    print()
    print("Time elapsed (my::lobpcg):      {}".format(t1))
    print("Time elapsed (scipy::eigh):     {}".format(t2))
    print("Time elapsed (scipy::lobpcg):   {}".format(t3))
    # print("Time elapsed (scipy::eigsh):    {}".format(t4))

    # n = 10
    # m = 2
    # M = np.random.rand(n, n)
    # M = M + M.T + n * np.eye(n)
    # ic(np.allclose(M, M.T))
    # ic(np.all(np.linalg.eigvals(M) > 0))
    # U = np.random.rand(n, m)
    # # V is M orthonormal external basis
    # V = np.random.rand(n, 2)
    # # find the orthonormal basis of M
    # # V[:, 0] = M @ U[:, 0]
    # # V[:, 0] = V[:, 0] / np.linalg.norm(V[:, 0])
    # # V[:, 1] = M @ U[:, 1]
    # # V[:, 1] = V[:, 1] - np.dot(V[:, 1], V[:, 0]) * V[:, 0]
    # # V[:, 1] = V[:, 1] / np.linalg.norm(V[:, 1])
    # # ic(V)
    # # ic(U)
    # U = svqbDrop(M, U)
    # ic(U.T @ M @ U)
    # U = orthoDrop(M, U, V)
    # ic(U)
    # ic(U.T @ M @ U)
    # ic(V.T @ M @ U)
    # generate Gaussian random matrix omega  with size n x m
