#include <mpi.h>  // include MPI
#include <omp.h>  // include openmp

#include <KokkosBlas1_axpby.hpp>  // include kokkoskernels
#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas1_mult.hpp>
#include <KokkosBlas1_reciprocal.hpp>
#include <KokkosBlas1_scal.hpp>
#include <KokkosBlas1_update.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <Kokkos_Core.hpp>  // include kokkos
#include <Kokkos_Core.hpp>
#include <array>  // std::array
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

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

template <typename T>
std::pair<View1D<T>, View2D<T>> lobpcg(View2D<T>& A, View2D<T>& X = None,
                                       View2D<T>& B = None, double tol = 1e-8,
                                       int maxiter = 200) {
  int n = X.extent(0);  // number of degrees of freedom
  int m = X.extent(1);  // number of eigenpairs to compute

  if (B.extent(0) == 0) {
    B = View2D<T>("B", n, n);
    Kokkos::deep_copy(B, Kokkos::IdentityMatrix<T, ExecSpace>(n));
  }

  // if M is None, set M = A^{-1}
  if (M.extent(0) == 0) {
    M = View2D<T>("M", n, n);

    // implement inverse of A
  }

  View2D<T> P("P", n, m);
  View1D<T> R("residual", m);
  View1D<T> lambda_("lambda_", m);
  View2D<T> Y("Y", n, m);
  View1D<T> XBX("XBX", m);
  View1D<T> XAX("XAX", m);
  View1D<T> mu("mu", m);
  View2D<T> BX("BX", n, m);
  View2D<T> AX("AX", n, m);

  Kokkos::deep_copy(P, 0.0);
  Kokkos::deep_copy(R, 1.0);
  Kokkos::deep_copy(XBX, 0.0);
  Kokkos::deep_copy(XAX, 0.0);
  Kokkos::deep_copy(mu, 0.0);

  int k = 0;
  residual = 1.0;
  while (k < maxiter && residual > tol) {
    // AX = A * X
    KokkosBlas::gemm("N", "N", 1.0, A, X, 0.0, AX);
    // BX = B * X
    KokkosBlas::gemm("N", "N", 1.0, B, X, 0.0, BX);

    // XBX = X^T * B * X
    KokkosBlas::gemv("T", 1.0, B, X, 0.0, XBX);

    // XAX = X^T * A * X
    KokkosBlas::gemv("T", 1.0, A, X, 0.0, XAX);

    // mu = XBX / XAX
    KokkosBlas::axpby(1.0, XBX, 0.0, mu);
    KokkosBlas::axpby(-1.0, XAX, 1.0, mu);
    KokkosBlas::reciprocal(mu);

    // R = BX - AX * mu
    KokkosBlas::axpby(1.0, BX, 0.0, R);
    KokkosBlas::axpby(-1.0, AX, mu, R);

    // W = M * R
    KokkosBlas::gemv("N", 1.0, M, R, 0.0, W);

    // Perform Rayleigh-Ritz procedure
    View2D<T> Z;
    if (k > 0) {
      // Z = hstack(W, X, P)
      Z = View2D<T>("Z", n, 3 * m);
      ZW = Kokkos::subview(Z, Kokkos::ALL(), Kokkos::make_pair(0, m));
      ZX = Kokkos::subview(Z, Kokkos::ALL(), Kokkos::make_pair(m, 2 * m));
      ZP = Kokkos::subview(Z, Kokkos::ALL(), Kokkos::make_pair(2 * m, 3 * m));
      Kokkos::deep_copy(ZW, W);
      Kokkos::deep_copy(ZX, X);
      Kokkos::deep_copy(ZP, P);
    } else {
      // Z = hstack(W, X)
      Z = View2D<T>("Z", n, 2 * m);
      ZW = Kokkos::subview(Z, Kokkos::ALL(), Kokkos::make_pair(0, m));
      ZX = Kokkos::subview(Z, Kokkos::ALL(), Kokkos::make_pair(m, 2 * m));
      Kokkos::deep_copy(ZW, W);
      Kokkos::deep_copy(ZX, X);
    }

    // gramA = Z^T * A * Z
    View2D<T> gramA("gramA", 3 * m, 3 * m);
    KokkosBlas::gemm("T", "N", 1.0, Z, A, 0.0, gramA);
    KokkosBlas::gemm("N", "N", 1.0, gramA, Z, 0.0, gramA);

    // gramB = Z^T * B * Z
    View2D<T> gramB("gramB", 3 * m, 3 * m);
    KokkosBlas::gemm("T", "N", 1.0, Z, B, 0.0, gramB);
    KokkosBlas::gemm("N", "N", 1.0, gramB, Z, 0.0, gramB);

    // lambda_, Y = eigh(gramA, gramB)
    

    k++;
  }

  return std::make_pair(lambda_, Y);
}