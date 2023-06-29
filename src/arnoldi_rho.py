import numpy as np

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
  
  
# def arnoldi_rho(A, k=10):
#     n = A.shape[0]

#     # if k > n and n > 20, then k = n
#     if k > n and n > 20:
#         k = n
    
#     # initialize the Arnoldi iteration
#     Q = np.zeros((n, k+1))
#     H = np.zeros((k+1, k))
#     Q[:, 0] = np.ones(n) / np.sqrt(n)
    
#     for j in range(k):
#         v = np.zeros(n)
#         for i in range(n):
#             v += A[i] * Q[i, j]
#         for i in range(j+1):
#             H[i, j] = np.dot(Q[:, i], v)
#             v = v - H[i, j] * Q[:, i]

#         # eps is machine precision
#         H[j+1, j] = np.linalg.norm(v) + np.finfo(float).eps
#         Q[:, j+1] = v / H[j+1, j]

#     # resize the matrix H to k x k and find the spectral radius of H
#     H = H[0:k, 0:k]

#     # find the spectral radius of H manually
#     x = np.ones(k)
#     rho = 0
#     for i in range(100):
#         x = np.dot(H, x)
#         rho = np.linalg.norm(x)
#         x /= rho
    
#     # compare the MSE of rho vs max_abs_eigs_A
#     max_abs_eigs_A = np.max(np.abs(np.linalg.eigvals(A)))
    
#     # print in 16 digits
#     print('max_abs_eigs_A =', max_abs_eigs_A)
#     print('arnoldi_rho_H  =', rho)
    
#     return rho

def arnoldi_rho(A, k, m):
    n = A.shape[0]

    # if k > n and n > 20, then k = n
    if k > n and n > 20:
        k = n
    
    # initialize the Arnoldi iteration
    Q = np.zeros((n, k+1))
    H = np.zeros((k+1, k))
    Q[:, 0] = np.ones(n) / np.sqrt(n)
    
    for j in range(k):
        v = np.zeros(n)
        for i in range(n):
            v += A[i] * Q[i, j]
        for i in range(j+1):
            H[i, j] = np.dot(Q[:, i], v)
            v = v - H[i, j] * Q[:, i]

        # eps is machine precision
        H[j+1, j] = np.linalg.norm(v) + np.finfo(float).eps
        Q[:, j+1] = v / H[j+1, j]

    # resize the matrix H to k x k and find the first m eigenpairs
    H = H[0:k, 0:k]

    eigenvalues = []
    eigenvectors = []
    for i in range(m):
        x = np.random.rand(k)
        for _ in range(100):
            x = np.dot(H, x)
            x /= np.linalg.norm(x)

        eigenvector = np.dot(Q[:, :k], x)
        eigenvalue = np.dot(eigenvector, np.dot(A, eigenvector))
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)

        # Deflate the matrix A
        A -= eigenvalue * np.outer(eigenvector, eigenvector)

    return eigenvalues, eigenvectors
  
if __name__ == "__main__":
  A = rand_symm_mat(n=10, eig_low=1.0, eig_high=10.0, nrepeat=1)
  spectral_radius, _ = arnoldi_rho(A, k=20, m =3)
  
  print('spectral_radius =', spectral_radius)
  
  eig = np.linalg.eigvals(A)
  print('eig =', eig)