#ifndef LOBPCG_HPP
#define LOBPCG_HPP

#include <KokkosBlas1_axpby.hpp>  // include kokkoskernels
#include <KokkosBlas1_dot.hpp>
#include <KokkosBlas1_mult.hpp>
#include <KokkosBlas1_nrm2.hpp>
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

#include "lapackage.hpp"
#include "utils.hpp"

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

namespace linalg {
/********************* declarations *********************/

template <typename T>
void lobpcg(T* Ap, T* Bp, int n, int m, T* wp, T* vp, T* Xp = nullptr,
            T* Mp = nullptr, double tol = 1e-8, int maxiter = 500) {
  View2D<T> A(Ap, n, n);
  View2D<T> B(Bp, n, n);
  View2D<T> X;
  View2D<T> M;

  if (Xp == nullptr) {
    X = View2D<T>("X", n, m);
    Kokkos::parallel_for(
        "lobpcg::initX",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, m}),
        KOKKOS_LAMBDA(int i, int j) {
          if (i == j) {
            X(i, j) = 1.0;
          } else {
            X(i, j) = 0.0;
          }
        });
  } else {
    X = View2D<T>(Xp, n, m);
  }

  if (Mp == nullptr) {
    M = View2D<T>("M", n, n);
    lapackage::inverse(A.data(), n, M.data());
  } else {
    M = View2D<T>(Mp, n, n);
  }

  tick("linalg::lobpcg");

  // printMat("M", M.data(), n, n);

  View2D<T> P("P", n, m);
  View2D<T> R("residual", n, m);
  View2D<T> BX("BX", n, m);
  View2D<T> AX("AX", n, m);
  View1D<T> mu("mu", m);
  View2D<T> W("W", n, m);

  int k = 0;
  T residual = 1.0;
  while (k < maxiter && residual > tol) {
    // AX = A * X
    KokkosBlas::gemm("N", "N", 1.0, A, X, 0.0, AX);
    // printMat("AX", AX.data(), n, m);
    //   // BX = B * X
    KokkosBlas::gemm("N", "N", 1.0, B, X, 0.0, BX);
    // printMat("BX", BX.data(), n, m);

    for (int i = 0; i < m; i++) {
      auto X_i = Kokkos::subview(X, Kokkos::ALL(), i);
      auto BX_i = Kokkos::subview(BX, Kokkos::ALL(), i);
      auto AX_i = Kokkos::subview(AX, Kokkos::ALL(), i);

      T xBx = KokkosBlas::dot(X_i, BX_i);
      T xAx = KokkosBlas::dot(X_i, AX_i);

      mu(i) = xBx / xAx;

      // R = BX - AX * mu
      auto R_i = Kokkos::subview(R, Kokkos::ALL(), i);
      KokkosBlas::update(1.0, BX_i, -mu(i), AX_i, 0.0, R_i);
    }

    // printMat("mu", mu.data(), 1, m);
    // printMat("R", R.data(), n, m);

    // W = M * R
    KokkosBlas::gemm("N", "N", 1.0, M, R, 0.0, W);
    // printMat("W", W.data(), n, m);

    // Perform Rayleigh-Ritz procedure
    View2D<T> Z;
    if (k > 0) {
      // Z = hstack([W, X, P])
      Z = View2D<T>("Z", n, 3 * m);
      // Kokkos::parallel_for(
      //     "lobpcg::initZ",
      //     Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, m}),
      //     KOKKOS_LAMBDA(int i, int j) {
      //       Z(i, j) = W(i, j);
      //       Z(i, j + m) = X(i, j);
      //       Z(i, j + 2 * m) = P(i, j);
      //     });
      auto ZW = Kokkos::subview(Z, Kokkos::ALL(), Kokkos::make_pair(0, m));
      auto ZX = Kokkos::subview(Z, Kokkos::ALL(), Kokkos::make_pair(m, 2 * m));
      auto ZP =
          Kokkos::subview(Z, Kokkos::ALL(), Kokkos::make_pair(2 * m, 3 * m));
      Kokkos::deep_copy(ZW, W);
      Kokkos::deep_copy(ZX, X);
      Kokkos::deep_copy(ZP, P);
    } else {
      // Z = hstack([W, X])
      Z = View2D<T>("Z", n, 2 * m);
      // Kokkos::parallel_for(
      //     "lobpcg::initZ",
      //     Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, m}),
      //     KOKKOS_LAMBDA(int i, int j) {
      //       Z(i, j) = W(i, j);
      //       Z(i, j + m) = X(i, j);
      //     });
      auto ZW = Kokkos::subview(Z, Kokkos::ALL(), Kokkos::make_pair(0, m));
      auto ZX = Kokkos::subview(Z, Kokkos::ALL(), Kokkos::make_pair(m, 2 * m));
      Kokkos::deep_copy(ZW, W);
      Kokkos::deep_copy(ZX, X);
    }

    // printMat("Z", Z.data(), n, 2 * m);

    // Compute symmetric gram matrices
    int xm = Z.extent(1);  // number of columns in Z
    View2D<T> gramA("gramA", xm, xm);
    View2D<T> gramB("gramB", xm, xm);
    View2D<T> ZA("ZA", xm, n);
    View2D<T> ZB("ZB", xm, n);

    // gramA = Z^T * A * Z
    KokkosBlas::gemm("T", "N", 1.0, Z, A, 0.0, ZA);
    KokkosBlas::gemm("N", "N", 1.0, ZA, Z, 0.0, gramA);

    // gramB = Z^T * B * Z
    KokkosBlas::gemm("T", "N", 1.0, Z, B, 0.0, ZB);
    KokkosBlas::gemm("N", "N", 1.0, ZB, Z, 0.0, gramB);

    // printMat("gramA", gramA.data(), 2 * m, 2 * m);
    // printMat("gramB", gramB.data(), 2 * m, 2 * m);

    // Compute eigenvalues and eigenvectors of reduced eigenvalue problem
    // View1D<T> evals("evals", m);
    View1D<T> Ycol("evecs_colmajor", xm * m);
    View2D<T> Y("evecs_rowmajor", xm, m);

    lapackage::sygvx<T>(gramA.data(), gramB.data(), xm, m, wp, Ycol.data());

    // Convert eigenvectors from column-major to row-major
    Kokkos::parallel_for(
        "lobpcg::convertEvecs",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {xm, m}),
        KOKKOS_LAMBDA(int i, int j) { Y(i, j) = Ycol(j * xm + i); });

    // printMat("evals", wp, m, 1);
    // printMat("evecs", Y.data(), xm, m);

    // Compute Ritz vectors Yw = Y[:m, :], Yx = Y[m : 2 * m, :]
    auto Yw = Kokkos::subview(Y, Kokkos::make_pair(0, m), Kokkos::ALL());
    auto Yx = Kokkos::subview(Y, Kokkos::make_pair(m, 2 * m), Kokkos::ALL());

    // printMat("Yw", Yw.data(), m, m);
    // printMat("Yx", Yx.data(), m, m);

    // Yp = Y[2 * m :, :]
    View2D<T> Yp("Yp", m, m);
    if (k > 0) {
      Yp = Kokkos::subview(Y, Kokkos::make_pair(2 * m, 3 * m), Kokkos::ALL());
    }

    // P = np.dot(P, Yp) + np.dot(W, Yw)
    // X = np.dot(P, Yp) + np.dot(W, Yw) + np.dot(X, Yx)
    View2D<T> XYx("XYx", n, m);
    View2D<T> PYp("PYp", n, m);
    View2D<T> WYw("WYw", n, m);
    KokkosBlas::gemm("N", "N", 1.0, X, Yx, 0.0, XYx);
    KokkosBlas::gemm("N", "N", 1.0, P, Yp, 0.0, PYp);
    KokkosBlas::gemm("N", "N", 1.0, W, Yw, 0.0, WYw);

    KokkosBlas::update(1.0, PYp, 1.0, WYw, 0.0, P);
    KokkosBlas::update(1.0, XYx, 1.0, P, 0.0, X);

    // printMat("X", X.data(), n, m);
    // printMat("P", P.data(), n, m);

    // residual = norm(R)
    // residual = 0.0;
    // for (int i = 0; i < m; i++){
    //   auto r = Kokkos::subview(R, Kokkos::ALL(), i);
    //   residual += KokkosBlas::nrm2(r);
    // }
    View1D<T> R_flat(R.data(), n * m);
    // printMat("R", R_flat.data(), n, m);

    residual = KokkosBlas::nrm2(R_flat);
    printf("Iteration %d, residual = %e\n", k, residual);

    k++;
  }

  // vp is the pointer to the eigenvector from X
  View2D<T> v(vp, n, m);
  Kokkos::deep_copy(v, X);
  // printMat("v", v.data(), n, m);

  tock("linalg::lobpcg");
}

}  // namespace linalg

#endif  // LOBPCG_HPP