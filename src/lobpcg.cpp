// #include <mpi.h>  // include MPI
// #include <omp.h>  // include openmp

#include <KokkosBlas1_axpby.hpp>  // include kokkoskernels
#include <KokkosBlas1_mult.hpp>
#include <KokkosBlas1_reciprocal.hpp>
#include <KokkosBlas1_scal.hpp>
#include <KokkosBlas1_update.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <Kokkos_Core.hpp>  // include kokkos
#include <Kokkos_Random.hpp>
#include <array>  // std::array
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "lapackage.h"
#include "utility.h"

#ifdef KOKKOS_ENABLE_CUDA
#define MemSpace Kokkos::CudaSpace
#endif

#ifndef MemSpace
#define MemSpace Kokkos::HostSpace
#endif

using ExecSpace = MemSpace::execution_space;
using RangePolicy = Kokkos::RangePolicy<ExecSpace>;

typedef Kokkos::DefaultExecutionSpace::array_layout Layout;

template <typename T>
using View1D = Kokkos::View<T*, Layout, ExecSpace>;
template <typename T>
using View2D = Kokkos::View<T**, Layout, ExecSpace>;

// template <typename T>
// std::pair<View1D<T>, View2D<T>> lobpcg(View2D<T>& A, View2D<T>& X = None,
//                                        View2D<T>& B = None, double tol =
//                                        1e-8, int maxiter = 200) {
//   int n = X.extent(0);  // number of degrees of freedom
//   int m = X.extent(1);  // number of eigenpairs to compute

//   if (B.extent(0) == 0) {
//     B = View2D<T>("B", n, n);
//     Kokkos::deep_copy(B, Kokkos::IdentityMatrix<T, ExecSpace>(n));
//   }

//   // if M is None, set M = A^{-1}
//   if (M.extent(0) == 0) {
//     M = View2D<T>("M", n, n);

//     // implement inverse of A
//   }

//   View2D<T> P("P", n, m);
//   View1D<T> R("residual", m);
//   View1D<T> lambda_("lambda_", m);
//   View2D<T> Y("Y", n, m);
//   View1D<T> XBX("XBX", m);
//   View1D<T> XAX("XAX", m);
//   View1D<T> mu("mu", m);
//   View2D<T> BX("BX", n, m);
//   View2D<T> AX("AX", n, m);

//   Kokkos::deep_copy(P, 0.0);
//   Kokkos::deep_copy(R, 1.0);
//   Kokkos::deep_copy(XBX, 0.0);
//   Kokkos::deep_copy(XAX, 0.0);
//   Kokkos::deep_copy(mu, 0.0);

//   int k = 0;
//   residual = 1.0;
//   while (k < maxiter && residual > tol) {
//     // AX = A * X
//     KokkosBlas::gemm("N", "N", 1.0, A, X, 0.0, AX);
//     // BX = B * X
//     KokkosBlas::gemm("N", "N", 1.0, B, X, 0.0, BX);

//     // XBX = X^T * B * X
//     KokkosBlas::gemv("T", 1.0, B, X, 0.0, XBX);

//     // XAX = X^T * A * X
//     KokkosBlas::gemv("T", 1.0, A, X, 0.0, XAX);

//     // mu = XBX / XAX
//     KokkosBlas::axpby(1.0, XBX, 0.0, mu);
//     KokkosBlas::axpby(-1.0, XAX, 1.0, mu);
//     KokkosBlas::reciprocal(mu);

//     // R = BX - AX * mu
//     KokkosBlas::axpby(1.0, BX, 0.0, R);
//     KokkosBlas::axpby(-1.0, AX, mu, R);

//     // W = M * R
//     KokkosBlas::gemv("N", 1.0, M, R, 0.0, W);

//     // Perform Rayleigh-Ritz procedure
//     View2D<T> Z;
//     if (k > 0) {
//       // Z = hstack(W, X, P)
//       Z = View2D<T>("Z", n, 3 * m);
//       ZW = Kokkos::subview(Z, Kokkos::ALL(), Kokkos::make_pair(0, m));
//       ZX = Kokkos::subview(Z, Kokkos::ALL(), Kokkos::make_pair(m, 2 * m));
//       ZP = Kokkos::subview(Z, Kokkos::ALL(), Kokkos::make_pair(2 * m, 3 *
//       m)); Kokkos::deep_copy(ZW, W); Kokkos::deep_copy(ZX, X);
//       Kokkos::deep_copy(ZP, P);
//     } else {
//       // Z = hstack(W, X)
//       Z = View2D<T>("Z", n, 2 * m);
//       ZW = Kokkos::subview(Z, Kokkos::ALL(), Kokkos::make_pair(0, m));
//       ZX = Kokkos::subview(Z, Kokkos::ALL(), Kokkos::make_pair(m, 2 * m));
//       Kokkos::deep_copy(ZW, W);
//       Kokkos::deep_copy(ZX, X);
//     }

//     // gramA = Z^T * A * Z
//     View2D<T> gramA("gramA", 3 * m, 3 * m);
//     KokkosBlas::gemm("T", "N", 1.0, Z, A, 0.0, gramA);
//     KokkosBlas::gemm("N", "N", 1.0, gramA, Z, 0.0, gramA);

//     // gramB = Z^T * B * Z
//     View2D<T> gramB("gramB", 3 * m, 3 * m);
//     KokkosBlas::gemm("T", "N", 1.0, Z, B, 0.0, gramB);
//     KokkosBlas::gemm("N", "N", 1.0, gramB, Z, 0.0, gramB);

//     // lambda_, Y = eigh(gramA, gramB)
//     auto result = dsygvx<T, 3 * m, m>(gramA, gramB);
//     View1D<T> lambda_ = result.eigenvalues;
//     View2D<T> Y = result.eigenvectors;

//     // Compute Ritz vectors Yw = Y[:m, :], Yx = Y[m : 2 * m, :]
//     View2D<T> Yw = Kokkos::subview(Y, Kokkos::make_pair(0, m),
//     Kokkos::ALL()); View2D<T> Yx =
//         Kokkos::subview(Y, Kokkos::make_pair(m, 2 * m), Kokkos::ALL());

//     if (k > 0) {
//       // Yp = Y[2 * m :, :]
//       View2D<T> Yp =
//           Kokkos::subview(Y, Kokkos::make_pair(2 * m, 3 * m), Kokkos::ALL());
//     } else {
//       // Yp = zeros((m, m))
//       View2D<T> Yp("Yp", m, m);
//       Kokkos::deep_copy(Yp, 0.0);
//     }

//     // X = np.dot(W, Yw) + np.dot(X, Yx) + np.dot(P, Yp)
//     // P = np.dot(W, Yw) + np.dot(P, Yp)
//     KokkosBlas::gemm("N", "N", 1.0, W, Yw, 0.0, X);
//     KokkosBlas::gemm("N", "N", 1.0, X, Yx, 1.0, X);
//     KokkosBlas::gemm("N", "N", 1.0, P, Yp, 0.0, P);
//     KokkosBlas::axpby(1.0, P, 1.0, X);
//     KokkosBlas::gemm("N", "N", 1.0, W, Yw, 1.0, P);

//     // residual = np.linalg.norm(R) / np.linalg.norm(A)
//     residual = KokkosBlas::nrm2(R) / KokkosBlas::nrm2(A);

//     printf("Iteration %d, residual = %e\n", k, residual);

//     k++;
//   }

//   return std::make_pair(lambda_, Y);
// }

int main() {
  typedef double T;
  int n = 10;
  int m = 10;
  int maxiter = 100;
  double tol = 1e-6;

  Kokkos::initialize();
  {
    View2D<T> A("A", n, n);
    View2D<T> B("B", n, n);
    View2D<T> I("W", n, n);
    View2D<T> M("M", n, n);
    View2D<T> X("X", n, m);

    Kokkos::Random_XorShift64_Pool<> rand_pool(1234);
    Kokkos::fill_random(A, rand_pool, -1.0, 1.0);
    Kokkos::fill_random(B, rand_pool, -1.0, 1.0);

    // A = A + A^T + n * np.eye(n), symmetric positive definite
    // B = B + B^T + n * np.eye(n), symmetric positive definite
    Kokkos::parallel_for(
        "construct_AB", n, KOKKOS_LAMBDA(const int i) {
          for (int j = 0; j < i + 1; j++) {
            A(i, j) = A(j, i);
            B(i, j) = B(j, i);
            if (i == j) {
              A(i, j) += n;
              B(i, j) += n;
            }
          }
        });

    // // Make A and B symmetric
    // KokkosBlas::gemm("N", "T", 1.0, A, A, 0.0, A);
    // KokkosBlas::gemm("N", "T", 1.0, B, B, 0.0, B);

    // // A = A + n * np.eye(n)
    // Kokkos::parallel_for(
    //     "fill_I", n, KOKKOS_LAMBDA(const int i) { I(i, i) = 1.0; });
    // printMat<View2D<T>>("I", I);
    // KokkosBlas::axpy(n, I, A);

    // M = inverse of A

    printMat<View2D<T>>("A", A);
    printMat<View2D<T>>("B", B);
    printMat<View2D<T>>("M", M);
    printMat<View2D<T>>("X", X);
  }
  Kokkos::finalize();
}