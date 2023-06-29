#include "dsygvx_lapack.h"

#include <iostream>
#include <vector>

// Solve the generalized symmetric eigenvalue problem (GSEP) Ax = ¦ËBx
// using LAPACK's dsygvx routine.
std::pair<std::vector<double>, std::vector<double>> dsygvx(
    std::vector<double>& A, std::vector<double>& B, int n, int m) {
  // Initialize variables
  int itype = 1;             // Ax = ¦ËBx
  char jobz = 'V';           // Compute eigenvalues and eigenvectors
  char range = 'I';          // Compute eigenvalues in il-th through iu-th range
  char uplo = 'U';           // Upper triangular part of A and B are stored
  int lda = n;               // Leading dimension of A
  int ldb = n;               // Leading dimension of B
  double vl = 0;             // Lower bound of eigenvalues not referenced
  double vu = 0;             // Upper bound of eigenvalues not referenced
  int il = 1;                // Index of smallest eigenvalue to compute
  int iu = m;                // Index of largest eigenvalue to compute
  double abstol = 0.0;       // Absolute tolerance
  std::vector<double> w(m);  // Eigenvalues
  std::vector<double> z(n * m);  // Eigenvectors
  int ldz = n;                   // Leading dimension of eigenvectors
  int lwork = 10 * n;            // Length of work array
  int info;                      // Return code

  // Initialize work arrays
  std::vector<double> work(lwork);
  std::vector<int> iwork(5 * n);
  std::vector<int> ifail(n);

  // Compute eigenvalues and eigenvectors
  dsygvx_(&itype, &jobz, &range, &uplo, &n, A.data(), &lda, B.data(), &ldb, &vl,
          &vu, &il, &iu, &abstol, &m, w.data(), z.data(), &ldz, work.data(),
          &lwork, iwork.data(), ifail.data(), &info);
  // Check for errors
  if (info != 0) {
    std::cout << "Error: dsygvx returned " << info << "\n";
    exit(1);
  }

  return std::make_pair(w, z);
}

int main() {
  // Example matrices A and B
  std::vector<double> A = {1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0};
  std::vector<double> B = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

  int n = 3;  // Dimension of the matrices
  int m = 2;  // Number of eigenpairs to compute

  // Solve the GSEP for the first m lowest eigenpairs
  std::pair<std::vector<double>, std::vector<double>> result =
      dsygvx(A, B, n, m);
  printf("Eigenvalues:\n");
  for (int i = 0; i < m; i++) {
    printf("%f\n", result.first[i]);
  }

  return 0;
}