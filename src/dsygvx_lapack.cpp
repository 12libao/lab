#include "lapackage.h"

#include <Accelerate/Accelerate.h>

#include <array>
#include <iostream>


extern "C" {
int dsygvx_(int* itype, char* jobz, char* range, char* uplo, int* n, double* a,
            int* lda, double* b, int* ldb, double* vl, double* vu, int* il,
            int* iu, double* abstol, int* m, double* w, double* z, int* ldz,
            double* work, int* lwork, int* iwork, int* ifail, int* info);
}

// Solve the generalized symmetric eigenvalue problem (GSEP) Ax = ¦ËBx
// using LAPACK's dsygvx routine.
template <typename T, std::size_t N, std::size_t M>
EigenPair<T, N, M> dsygvx(const std::array<T, N * N>& A,
                          const std::array<T, N * N>& B) {
  // Initialize variables
  const int n = N;  // Order of matrices A and B
  const int m = M;  // Number of eigenvalues to compute

  int itype = 1;             // Ax = ¦ËBx
  char jobz = 'V';           // Compute eigenvalues and eigenvectors
  char range = 'I';          // Compute in il-th through iu-th range
  char uplo = 'U';           // Upper triangular part of A and B are stored
  int lda = n;               // Leading dimension of A
  int ldb = n;               // Leading dimension of B
  T vl = 0;                  // Lower bound of eigenvalues not referenced
  T vu = 0;                  // Upper bound of eigenvalues not referenced
  int il = 1;                // Index of smallest eigenvalue to compute
  int iu = m;                // Index of largest eigenvalue to compute
  double abstol = 0.0;       // Absolute tolerance
  std::array<T, m> w{};      // Eigenvalues
  std::array<T, n * m> z{};  // Eigenvectors
  int ldz = n;               // Leading dimension of eigenvectors
  const int lwork = 10 * n;  // Length of work array
  int info;                  // Return code

  // Initialize work arrays
  std::array<T, lwork> work{};     // Work array
  std::array<int, 5 * n> iwork{};  // Integer work array
  std::array<int, n> ifail{};      // Failure index array

  // Copy n, m, and lwork to variables that can be passed by reference
  int nValue = n;
  int mValue = m;
  int lValue = lwork;

  // Compute eigenvalues and eigenvectors
  dsygvx_(&itype, &jobz, &range, &uplo, &nValue, const_cast<T*>(A.data()), &lda,
          const_cast<T*>(B.data()), &ldb, &vl, &vu, &il, &iu, &abstol, &mValue,
          w.data(), z.data(), &ldz, work.data(), &lValue, iwork.data(),
          ifail.data(), &info);

  // Check for errors
  if (info != 0) {
    throw std::runtime_error("Error: dsygvx returned " + std::to_string(info));
  }

  return EigenPair<T, N, M>{w, z};
}

int main() {
  std::array<double, 9> A = {1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0};
  std::array<double, 9> B = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

  constexpr size_t n = 3;
  constexpr size_t m = 2;

  auto result = dsygvx<double, n, m>(A, B);

  std::cout << "Eigenvalues:\n";
  for (double eigenvalue : result.eigenvalues) {
    std::cout << eigenvalue << "\n";
  }

  std::cout << "Eigenvectors:\n";
  for (size_t i = 0; i < m; ++i) {
    std::cout << "Eigenvalue " << i << ":\n";
    for (size_t j = 0; j < n; ++j) {
      std::cout << result.eigenvectors[i * n + j] << "\n";
    }
  }
  return 0;
}