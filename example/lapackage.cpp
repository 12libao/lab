#include "lapackage.h"

#include <lapacke.h>

#include <KokkosBlas1_abs.hpp>
#include <KokkosBlas1_axpby.hpp>  // include kokkoskernels
#include <KokkosBlas1_mult.hpp>
#include <KokkosBlas1_nrm2.hpp>
#include <KokkosBlas1_reciprocal.hpp>
#include <KokkosBlas1_scal.hpp>
#include <KokkosBlas1_update.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <Kokkos_Core.hpp>  // include kokkos
#include <array>
#include <cstdio>

// #include "utility.h"

// extern "C" {
// // // LU decomoposition of a general matrix
// // void dgetrf_(int* M, int* N, double* A, int* lda, int* IPIV, int*
// // INFO);

// // // generate inverse of a matrix given its LU decomposition
// // void LAPACK_dgetri(int* N, double* A, int* lda, int* IPIV, double* WORK,
// // int* lwork,
// //             int* INFO);

// void LAPACK_dsygvx(int* itype, char* jobz, char* range, char* uplo, int* n,
//                     double* a, int* lda, double* b, int* ldb, double* vl,
//                     double* vu, int* il, int* iu, double* abstol, int* m,
//                     double* w, double* z, int* ldz, double* work, int* lwork,
//                     int* iwork, int* ifail, int* info);
// }



namespace lapackage {
// template <typename T>
// void inverse(T* A, int N) {
//   int* IPIV = new int[N];
//   int LWORK = N * N;
//   double* WORK = new double[LWORK];
//   int INFO;

//   lapack_getrf(&N, &N, A, &N, IPIV, &INFO);
//   lapack_getri(&N, A, &N, IPIV, WORK, &LWORK, &INFO);

//   delete[] IPIV;
//   delete[] WORK;
// }

// Solve the generalized symmetric eigenvalue problem (GSEP) Ax = ¦ËBx
// using LAPACK's dsygvx routine.

}  // namespace lapackage

// int main() {
//   typedef double T;
//   int maxiter = 100;
//   double tol = 1e-6;

//   Kokkos::initialize();
//   {
//     Kokkos::View<double**, Kokkos::HostSpace> A("A", 2, 2);
//     A(0, 0) = 1;
//     A(0, 1) = 2;
//     A(1, 0) = 3;
//     A(1, 1) = 4;

//     // convert to C array
//     double* A_ptr = A.data();

//     inverse(A_ptr, 2);
//     printf("%f %f\n", A(0, 0), A(0, 1));
//     printf("%f %f\n", A(1, 0), A(1, 1));

//     constexpr int n = 3;
//     constexpr int m = 2;

//     std::array<T, n* n> A2 = {1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0};
//     std::array<T, n* n> B2 = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
//     std::array<T, m> eigenvalues2;
//     std::array<T, n * m> eigenvectors2;

//     sygvx<double>(A2.data(), B2.data(), n, m, eigenvalues2.data(),
//                   eigenvectors2.data());

//     for (int i = 0; i < m; ++i) {
//       printf("%f\n", eigenvalues2[i]);
//     }
//     printf("\n");

//     for (T eigenvector : eigenvectors2) {
//       printf("%f ", eigenvector);
//     }
//     printf("\n");

//     // compute res = A * x - ¦Ë * B * x
//     std::array<T, n * m> res;
//     for (int i = 0; i < n; ++i) {
//       for (int j = 0; j < m; j++) {
//         res[i * m + j] = 0.0;
//         for (int k = 0; k < n; ++k) {
//           res[i * m + j] += A2[i * n + k] * eigenvectors2[k * m + j];
//         }
//         res[i * m + j] -= eigenvalues2[j] * eigenvectors2[i * m + j];
//       }
//     }

//     // compute norm of res
//     T norm = 0.0;
//     for (int i = 0; i < n; ++i) {
//       for (int j = 0; j < m; j++) {
//         norm += res[i * m + j] * res[i * m + j];
//       }
//     }
//     norm = std::sqrt(norm);
//     printf("norm = %f\n", norm);

//     Kokkos::View<T**, Kokkos::HostSpace> A3("A", n, n);
//     Kokkos::View<T**, Kokkos::HostSpace> B3("B", n, n);
//     Kokkos::View<T*, Kokkos::HostSpace> eigenvalues("eigenvalues", m);
//     Kokkos::View<T**, Kokkos::LayoutLeft, Kokkos::HostSpace> eigenvectors(
//         "eigenvectors", n, m);

//     for (int i = 0; i < n; ++i) {
//       for (int j = 0; j < n; j++) {
//         if (i == j) {
//           A3(i, j) = i + 1;
//           B3(i, j) = 1.0;
//         } else {
//           A3(i, j) = 0.0;
//           B3(i, j) = 0.0;
//         }
//       }
//     }

//     // auto result = dsygvx<double, n, m>(A2, B2);
//     sygvx<double>(A3.data(), B3.data(), n, m, eigenvalues.data(),
//                   eigenvectors.data());

//     std::cout << "Eigenvalues:\n";
//     for (int i = 0; i < m; ++i) {
//       std::cout << eigenvalues(i) << "\n";
//     }

//     std::cout << "Eigenvectors:\n";
//     for (int i = 0; i < m; ++i) {
//       std::cout << "Eigenvalue " << i << ":\n";
//       for (int j = 0; j < n; ++j) {
//         std::cout << eigenvectors(j, i) << "\n";
//       }
//     }

//     // check that A * x = lambda * B * x
//     Kokkos::View<T**, Kokkos::HostSpace> residual("residual", n, m);
//     KokkosBlas::gemm("N", "N", 1.0, B3, eigenvectors, 0.0, residual);
//     KokkosBlas::scal(residual, eigenvalues, residual);
//     KokkosBlas::gemm("N", "N", -1.0, A3, eigenvectors, 1.0, residual);

//     T residual_norm = 0.0;

//     for (int i = 0; i < m; ++i) {
//       T norm = 0.0;
//       for (int j = 0; j < n; ++j) {
//         norm += residual(j, i) * residual(j, i);
//       }
//       residual_norm += std::sqrt(norm);
//     }

//     std::cout << "Residual norm: " << residual_norm << "\n";

//   }
//   Kokkos::finalize();
// }