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


def syevx3x3_analytical(A):
    eigvals = np.zeros(3)
    p1 = A[0, 1] ** 2 + A[0, 2] ** 2 + A[1, 2] ** 2
    if p1 == 0:
        eigvals[0] = A[0, 0]
        eigvals[1] = A[1, 1]
        eigvals[2] = A[2, 2]
    else:
        q = np.trace(A) / 3
        p2 = (A[0, 0] - q) ** 2 + (A[1, 1] - q) ** 2 + (A[2, 2] - q) ** 2 + 2 * p1
        p = np.sqrt(p2 / 6)
        temp = (1 / p) * (A - q * np.eye(3))
        r = np.linalg.det(temp) / 2

        if r <= -1 + 1e-10:
            phi = np.pi / 3
        elif r >= 1 - 1e-10:
            phi = 0
        else:
            phi = np.arccos(r) / 3

        eigvals[0] = q + 2 * p * np.cos(phi + (2 * np.pi / 3))
        eigvals[1] = q + 2 * p * np.cos(phi - (2 * np.pi / 3))
        eigvals[2] = 3 * q - eigvals[0] - eigvals[1]

    # find corresponding eigenvectors
    v1 = A - eigvals[0] * np.eye(3)
    v2 = A - eigvals[1] * np.eye(3)
    v3 = A - eigvals[2] * np.eye(3)

    v = np.zeros((3, 3))
    v[:, 0] = v2 @ v3[:, 1]
    v[:, 1] = v3 @ v1[:, 2]
    v[:, 2] = v1 @ v2[:, 0]

    v[:, 0] = v[:, 0] / np.linalg.norm(v[:, 0])
    v[:, 1] = v[:, 1] / np.linalg.norm(v[:, 1])
    v[:, 2] = v[:, 2] / np.linalg.norm(v[:, 2])

    # error1 = np.linalg.norm(v1 @ v[:, 0])
    # error2 = np.linalg.norm(v2 @ v[:, 1])
    # error3 = np.linalg.norm(v3 @ v[:, 2])
    # ic(error1, error2, error3)

    return eigvals, v


def sygvx3x3(A, B):
    Ob, Cb = syevx3x3_analytical(B)
    phi_B = Cb @ np.linalg.inv(np.diag(Ob ** (1 / 2) + 1e-15))
    Oa, Ca = syevx3x3_analytical(phi_B.T @ A @ phi_B)
    C_ab = phi_B @ Ca
    return Oa, C_ab


def syevx2x2_analytical(A):
    a00 = A[0, 0]
    a01 = A[0, 1]
    a11 = A[1, 1]

    trA = a00 + a11
    detA = (a00 * a11) - (a01 * a01)
    gapA = np.sqrt(trA * trA - 4 * detA)

    eigvals = np.zeros(2)
    eigvals[0] = (trA - gapA) / 2
    eigvals[1] = (trA + gapA) / 2

    v = np.zeros((2, 2))
    v[0, 0] = 1 / np.sqrt(1 + (a00 - eigvals[0]) ** 2 / a01**2)
    v[1, 0] = (eigvals[0] - a00) / a01 * v[0, 0]

    v[0, 1] = 1 / np.sqrt(1 + (a00 - eigvals[1]) ** 2 / a01**2)
    v[1, 1] = (eigvals[1] - a00) / a01 * v[0, 1]

    # error1 = np.linalg.norm(A @ v[:, 0] - eigvals[0] * v[:, 0])
    # error2 = np.linalg.norm(A @ v[:, 1] - eigvals[1] * v[:, 1])
    # ic(error1, error2)

    return eigvals, v


def sygvx2x2(A, B):
    Ob, Cb = syevx2x2_analytical(B)
    phi_B = Cb @ np.linalg.inv(np.diag(Ob ** (1 / 2) + 1e-15))
    Oa, Ca = syevx2x2_analytical(phi_B.T @ A @ phi_B)
    C_ab = phi_B @ Ca
    return Oa, C_ab


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
    return C, eigval


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
    N, m0 = X.shape
    m1 = int(np.ceil(1 * m0))
    m = m0 + m1
    ic(m0, m1)

    if B is None:
        B = np.eye(N)

    X = np.eye(N, m)
    XAX = A[:m, :m]
    XBX = B[:m, :m]
    # C, O = RayleighRitz3(XAX, XBX, m)
    O, C = eigh(XAX, XBX)

    X = X @ C
    # X = X / np.linalg.norm(X)

    R = A @ X - B @ X @ np.diag(O)
    P = np.zeros((N, m))

    non_convergent_indx = np.arange(m)

    # M = np.linalg.inv(A)
    AX = A @ X
    AP = np.zeros((N, m))
    BX = B @ X
    BP = np.zeros((N, m))
    
    # X = X / np.linalg.norm(X, axis=0)
    # AX = AX / np.linalg.norm(X, axis=0)
    # BX = BX / np.linalg.norm(X, axis=0)
    # R = R / np.linalg.norm(R, axis=0)

    # # find eigenvalues and eigenvectors for B
    if B is not None:
        eigs, eigv = eigh(B, subset_by_index=[0, 0])
    ic(eigs.shape, eigv.shape)
    M = eigv @ np.diag(eigs ** (-1)) @ eigv.T
    for k in range(maxiter):
        W = R
        MW = M @ W
        AMW = A @ MW
        AW = M.T @ AMW
        BW = B @ R
        # AW = A @ W
        # BW = B @ W
        # if k > 0:
        #     W = ortho(B, R, np.hstack((X, P)))
        # else:
        #     W = ortho(B, R, X)
        # ic(P)
        # W = W / np.linalg.norm(W)
        for i in non_convergent_indx:
            Xi_norm = np.linalg.norm(X[:, i])
            Wi_norm = np.linalg.norm(W[:, i])
            # ic(Xi_norm, Wi_norm)
            # Xi_norm = np.max(np.abs(X[:, i]))
            # Wi_norm = np.max(np.abs(W[:, i]))

            W[:, i] = W[:, i] / Wi_norm
            X[:, i] = X[:, i] / Xi_norm

            AX[:, i] = AX[:, i] / Xi_norm
            AW[:, i] = AW[:, i] / Wi_norm

            BX[:, i] = BX[:, i] / Xi_norm
            BW[:, i] = BW[:, i] / Wi_norm

            if k > 0:
                Pi_norm = np.linalg.norm(P[:, i])
                # Pi_norm = np.max(np.abs(P[:, i]))
                P[:, i] = P[:, i] / Pi_norm
                AP[:, i] = AP[:, i] / Pi_norm
                BP[:, i] = BP[:, i] / Pi_norm
    
        if k > 0:
            # P = P / np.linalg.norm(P)
            S = np.hstack((X, W, P))
            AS = np.hstack((AX, AW, AP))
            BS = np.hstack((BX, BW, BP))
        else:
            S = np.hstack((X, W))
            AS = np.hstack((AX, AW))
            BS = np.hstack((BX, BW))

        # S = S / np.linalg.norm(S)

        SAS = S.T @ AS
        SBS = S.T @ BS

        # ic(SAS)
        # ic(SBS)
        # ic(S)
        # ic(AS)
        # ic(BS)

        O, C = eigh(SAS, SBS, subset_by_index=[0, m - 1])
        # C, O = RayleighRitz3(SAS, SBS, m)
        # ic(O)
        # ic(C)
        X = S @ C
        AX = AS @ C
        BX = BS @ C
        
        P = S[:, m:] @ C[m:, :]
        AP = AS[:, m:] @ C[m:, :]
        BP = BS[:, m:] @ C[m:, :]
        
        # ic(X)
        # ic(W)
        # ic(P)
        # ic(AX)
        # ic(AW)
        # ic(AP)
        # ic(BX)
        # ic(BW)
        # ic(BP)
        
        R = AX - BX @ np.diag(O)
        # ic(R)

        res_max = 0
        non_convergent_indx = []

        for i in range(m0):
            residual = np.linalg.norm(R[:, i]) / np.linalg.norm(
                AX[:, i] + BX[:, i] * O[i]
            )
            # ic(residual)
            if residual > res_max:
                res_max = residual
            if residual > tol:
                non_convergent_indx.append(i)

        counter = m0 - len(non_convergent_indx)
        ic(k, counter, res_max)

        if res_max < tol:
            break

        for i in range(m0, m):
            residual = np.linalg.norm(R[:, i]) / np.linalg.norm(
                AX[:, i] + BX[:, i] * O[i]
            )
            if residual > tol:
                non_convergent_indx.append(i)

    return O[:m0], X[:, :m0]


def lobpcg4(A, X, B=None, M=None, tol=1e-8, maxiter=500):
    N, m0 = X.shape
    m1 = int(np.ceil(0.3 * m0))
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

    X = X @ C
    # X = X / np.linalg.norm(X)
    R = A @ X - B @ X @ np.diag(O)
    P = np.zeros((N, m))

    non_convergent_indx = np.arange(m)

    AX = A @ X
    AW = A @ R
    AP = A @ P
    BX = B @ X
    BW = B @ R
    BP = B @ P

    M = np.eye(N)
    # find eigenvalues and eigenvectors for B
    if B is not None:
        eigs, eigv = eigh(B)
    M = eigv @ np.diag(eigs ** (-1)) @ eigv.T 
    for k in range(maxiter):
        W = R @ M
        AW = A @ W @ M
        BW = B @ W @ M

        for i in non_convergent_indx:
            # Xi_norm = np.linalg.norm(X[:, i])
            # Wi_norm = np.linalg.norm(W[:, i])
            # Xi_norm = np.max(np.abs(X[:, i]))
            # Wi_norm = np.max(np.abs(W[:, i]))

            # W[:, i] = W[:, i] / Wi_norm
            # X[:, i] = X[:, i] / Xi_norm

            # AX[:, i] = AX[:, i] / Xi_norm
            # AW[:, i] = AW[:, i] / Wi_norm

            # BX[:, i] = BX[:, i] / Xi_norm
            # BW[:, i] = BW[:, i] / Wi_norm

            # if k > 0:
            #     Pi_norm = np.linalg.norm(P[:, i])
            #     Pi_norm = np.max(np.abs(P[:, i]))
            #     P[:, i] = P[:, i] / Pi_norm
            #     AP[:, i] = AP[:, i] / Pi_norm
            #     BP[:, i] = BP[:, i] / Pi_norm

            S = np.vstack((X[:, i], W[:, i], P[:, i])).T
            AS = np.vstack((AX[:, i], AW[:, i], AP[:, i])).T
            BS = np.vstack((BX[:, i], BW[:, i], BP[:, i])).T

            # S = S / np.linalg.norm(S, axis=0)

            SAS = S.T @ AS
            SBS = S.T @ BS

            if k > 0:
                O, C = eigh(SAS, SBS, subset_by_index=[0, 0])
                Oa, Ca = sygvx3x3(SAS, SBS)
            else:
                O, C = eigh(SAS[:2, :2], SBS[:2, :2], subset_by_index=[0, 0])
                Oa, Ca = sygvx2x2(SAS[:2, :2], SBS[:2, :2])

            ic(np.allclose(np.abs(C[:, 0]), np.abs(Ca[:, 0]), atol=1e-8))

            O = Oa[0]
            C = Ca

            if k > 0:
                P[:, i] = C[1, 0] * W[:, i] + C[2, 0] * P[:, i]
                AP[:, i] = C[1, 0] * AW[:, i] + C[2, 0] * AP[:, i]
                BP[:, i] = C[1, 0] * BW[:, i] + C[2, 0] * BP[:, i]
            else:
                P[:, i] = C[1, 0] * W[:, i]
                AP[:, i] = C[1, 0] * AW[:, i]
                BP[:, i] = C[1, 0] * BW[:, i]

            X[:, i] = P[:, i] + C[0, 0] * X[:, i]
            AX[:, i] = AP[:, i] + C[0, 0] * AX[:, i]
            BX[:, i] = BP[:, i] + C[0, 0] * BX[:, i]

        XAX = X.T @ AX
        XBX = X.T @ BX
        O, C = eigh(XAX, XBX)

        X = X @ C
        P = P @ C
        AX = AX @ C
        BX = BX @ C
        AP = AP @ C
        BP = BP @ C
        R = BX @ np.diag(O) - AX

        res_max = 0
        non_convergent_indx = []

        for i in range(m0):
            # residual = np.linalg.norm(R[:, i]) / (
            #     (Anorm + Bnorm * O[i]) * np.linalg.norm(X[:, i])
            # )
            residual = np.linalg.norm(R[:, i]) / np.linalg.norm(
                AX[:, i] + BX[:, i] * O[i]
            )
            # ic(residual)
            if residual > res_max:
                res_max = residual
            if residual > tol:
                non_convergent_indx.append(i)

        counter = m0 - len(non_convergent_indx)
        ic(k, counter, res_max)

        if res_max < tol:
            break

        for i in range(m0, m):
            # residual = np.linalg.norm(R[:, i]) / (
            #     (Anorm + Bnorm * np.abs(O[i])) * np.linalg.norm(X[:, i])
            # )
            residual = np.linalg.norm(R[:, i]) / np.linalg.norm(
                AX[:, i] + BX[:, i] * O[i]
            )
            if residual > tol:
                non_convergent_indx.append(i)
    # ic(X[:, :m0])
    return O[:m0], X[:, :m0]



if __name__ == "__main__":
    n = 10000  # Size of the matrix
    m = 100  # Number of desired eigenpairs
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
    lam1, vec1 = lobpcg3(A, X, B, M=M)
    end = time.time()
    t1 = end - start
    ic("my::lobpcg", t1)

    start = time.time()
    lam2, vec2 = eigh(A, B, subset_by_index=[0, m])
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
    print("Eigenvalues (scipy::eigh):     ", lam2)
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
    #     # XAX = O[0].item()
    #     XAX = X[:, i].T @ AX
    #     XAW = X[:, i].T @ AW
    #     XAP = X[:, i].T @ AP
    #     WAW = W[:, i].T @ AW
    #     WAP = W[:, i].T @ AP
    #     PAP = P[:, i].T @ AP

    #     XBX = X[:, i].T @ BX
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
    
#########################################

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
