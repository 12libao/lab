#ifndef LOBPCG_HPP
#define LOBPCG_HPP

#include <KokkosBlas.hpp>
// #include <KokkosBlas1_axpby.hpp>  // include kokkoskernels
// #include <KokkosBlas1_dot.hpp>
// #include <KokkosBlas1_mult.hpp>
// #include <KokkosBlas1_nrm2.hpp>
// #include <KokkosBlas1_reciprocal.hpp>
// #include <KokkosBlas1_scal.hpp>
// #include <KokkosBlas1_update.hpp>
// #include <KokkosBlas2_gemv.hpp>
// #include <KokkosBlas3_gemm.hpp>
#include <Kokkos_Core.hpp>  // include kokkos
#include <Kokkos_Random.hpp>
#include <array>  // std::array
#include <cmath>
#include <cstdlib>
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
            T* Mp = nullptr, double tol = 1e-8, int maxiter = 500,
            bool verbose = true) {
  View2D<T> A(Ap, n, n);
  View2D<T> B;
  View2D<T> X("X", n, m);
  View2D<T> M;
  View2D<T> XAX("XAX", m, m);
  View2D<T> XBX("XBX", m, m);

  View2D<T> AX("AX", n, m);
  View2D<T> BX("BX", n, m);

  View1D<T> w(wp, m);
  View2D<T> v("eigenvectors column major", m, m);

  if (m > int(ceil(n * 0.3))) {
    printf("m is larger than 30%% of m, use Lapack::sygvx instead\n");
    return;
  }

  if (Bp == nullptr) {
    B = View2D<T>("B", n, n);
    Kokkos::deep_copy(B, 0.0);
    Kokkos::parallel_for(
        "lobpcg::setB",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, n}),
        KOKKOS_LAMBDA(int i, int j) { B(i, j) = (i == j) ? 1.0 : 0.0; });
  } else {
    B = View2D<T>(Bp, n, n);
  }

  /* Compute XAX = X.T * A * X, XBX = X.T * B * X */
  if (Xp == nullptr) {
    //  X = eye(n, m) -> XAX = A[:m, :m], XBX = B[:m, :m]
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        XAX(i, j) = A(i, j);
        XBX(i, j) = B(i, j);
      }
    }
  } else {
    X = View2D<T>(Xp, n, m);
    KokkosBlas::gemm("N", "N", 1.0, A, X, 0.0, AX);    // AX = A * X
    KokkosBlas::gemm("N", "N", 1.0, B, X, 0.0, BX);    // BX = B * X
    KokkosBlas::gemm("T", "N", 1.0, X, AX, 0.0, XAX);  // XAX = X.T * AX
    KokkosBlas::gemm("T", "N", 1.0, X, BX, 0.0, XBX);  // XBX = X.T * BX
  }

  /* Solve generalized eigenvalue problem to get initial eigenvectors */
  lapackage::sygvx<T>(XAX.data(), XBX.data(), m, m, w.data(), v.data());

  /* Compute: AX = A * X, BX = B * X */
  if (Xp == nullptr) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        X(i, j) = v(j, i);  // X = eye(n, m) -> X = hstack(v, 0)
      }
    }

    auto A_nm = Kokkos::subview(A, Kokkos::ALL(), Kokkos::make_pair(0, m));
    auto B_nm = Kokkos::subview(B, Kokkos::ALL(), Kokkos::make_pair(0, m));
    KokkosBlas::gemm("N", "T", 1.0, A_nm, v, 0.0, AX);  // AX = A * v
    KokkosBlas::gemm("N", "T", 1.0, B_nm, v, 0.0, BX);  // BX = B * v
  } else {
    View2D<T> X0("last X", n, m);
    Kokkos::deep_copy(X0, X);
    KokkosBlas::gemm("N", "T", 1.0, X, v, 0.0, X);   // X = X * v
    KokkosBlas::gemm("N", "N", 1.0, A, X, 0.0, AX);  // AX = A * X
    KokkosBlas::gemm("N", "N", 1.0, B, X, 0.0, BX);  // BX = B * X
  }

  /* Compute residual: R = AX - BX * w, A_norm, B_norm */
  T A_norm = 0.0;
  T B_norm = 0.0;
  T X_norm = 0.0;
  View2D<T> R("residual", n, m);
  for (int i = 0; i < m; i++) {
    auto x = Kokkos::subview(X, Kokkos::ALL(), i);
    auto r = Kokkos::subview(R, Kokkos::ALL(), i);
    auto Ax = Kokkos::subview(AX, Kokkos::ALL(), i);
    auto Bx = Kokkos::subview(BX, Kokkos::ALL(), i);

    A_norm += KokkosBlas::nrm2_squared(Ax);  // A_norm += Ax^2
    B_norm += KokkosBlas::nrm2_squared(Bx);  // B_norm += Bx^2
    X_norm += KokkosBlas::nrm2_squared(x);   // X_norm += x^2

    KokkosBlas::update(1.0, Ax, -w(i), Bx, 0.0, r);  // r = Ax - Bx * w(i)
  }

  A_norm = sqrt(A_norm / X_norm);
  B_norm = sqrt(B_norm / X_norm);

  View2D<T> P("P", n, m);
  View1D<T> mu("mu", m);
  View2D<T> W("W", n, m);

  int k = 0;
  T residual = 1.0;
  while (k < maxiter && residual > tol) {
    if (Mp != nullptr) {  // W = M * R, preconditioning, normally M = A^-1
      KokkosBlas::gemm("N", "N", 1.0, M, R, 0.0, W);
    } else {  // without preconditioning
      Kokkos::deep_copy(W, R);
    }
    // printMat("W", W.data(), n, m);

    /* Perform Rayleigh-Ritz procedure */
    /* Z = [W, X, P] or [W, X] */
    View2D<T> Z;
    if (k > 0) {
      Z = View2D<T>("Z", n, 3 * m);
      Kokkos::parallel_for(
          "S = [X, W, P]",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, m}),
          KOKKOS_LAMBDA(int i, int j) {
            Z(i, j) = W(i, j);
            Z(i, j + m) = X(i, j);
            Z(i, j + 2 * m) = P(i, j);
          });
    } else {
      Z = View2D<T>("Z", n, 2 * m);
      Kokkos::parallel_for(
          "S = [X, W]",
          Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, m}),
          KOKKOS_LAMBDA(int i, int j) {
            Z(i, j) = W(i, j);
            Z(i, j + m) = X(i, j);
          });
    }

    // printMat("Z", Z.data(), n, 2 * m);

    /* Compute symmetric gram matrices */
    int xm = Z.extent(1);  // number of columns in Z
    View2D<T> gramA("gramA", xm, xm);
    View2D<T> gramB("gramB", xm, xm);
    View2D<T> ZA("ZA", xm, n);
    View2D<T> ZB("ZB", xm, n);

    /* gramA = Z^T * A * Z */
    KokkosBlas::gemm("T", "N", 1.0, Z, A, 0.0, ZA);
    KokkosBlas::gemm("N", "N", 1.0, ZA, Z, 0.0, gramA);

    /* gramB = Z^T * B * Z */
    KokkosBlas::gemm("T", "N", 1.0, Z, B, 0.0, ZB);
    KokkosBlas::gemm("N", "N", 1.0, ZB, Z, 0.0, gramB);

    // printMat("gramA", gramA.data(), 2 * m, 2 * m);
    // printMat("gramB", gramB.data(), 2 * m, 2 * m);

    /* Compute eigenvalues and eigenvectors of reduced eigenvalue problem */
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

    /* Compute Ritz vectors */
    /* Yw = Y[:m, :], Yx = Y[m : 2 * m, :], Yp = Y[2 * m :, :]*/
    auto Yw = Kokkos::subview(Y, Kokkos::make_pair(0, m), Kokkos::ALL());
    auto Yx = Kokkos::subview(Y, Kokkos::make_pair(m, 2 * m), Kokkos::ALL());

    // printMat("Yw", Yw.data(), m, m);
    // printMat("Yx", Yx.data(), m, m);

    View2D<T> Yp;
    if (k > 0) {
      Yp = Kokkos::subview(Y, Kokkos::make_pair(2 * m, 3 * m), Kokkos::ALL());
    } else {
      Yp = View2D<T>("Yp", m, m);
    }

    /* P = P * Yp + W * Yw */
    /* X = P * Yp + W * Yw + X * Yx */
    View2D<T> XYx("XYx", n, m);
    View2D<T> PYp("PYp", n, m);
    View2D<T> WYw("WYw", n, m);
    KokkosBlas::gemm("N", "N", 1.0, X, Yx, 0.0, XYx);
    KokkosBlas::gemm("N", "N", 1.0, P, Yp, 0.0, PYp);
    KokkosBlas::gemm("N", "N", 1.0, W, Yw, 0.0, WYw);

    KokkosBlas::update(1.0, PYp, 1.0, WYw, 0.0, P);
    KokkosBlas::update(1.0, XYx, 1.0, P, 0.0, X);

    /* Compute residual */
    KokkosBlas::gemm("N", "N", 1.0, A, X, 0.0, AX);  // AX = A * X
    KokkosBlas::gemm("N", "N", 1.0, B, X, 0.0, BX);  // BX = B * X
    for (int i = 0; i < m; i++) {
      auto x = Kokkos::subview(X, Kokkos::ALL(), i);
      auto Bx = Kokkos::subview(BX, Kokkos::ALL(), i);
      auto Ax = Kokkos::subview(AX, Kokkos::ALL(), i);
      auto r = Kokkos::subview(R, Kokkos::ALL(), i);

      KokkosBlas::update(1.0, Ax, -w(i), Bx, 0.0, r);  // r = Ax - Bx * w(i)
    }

    // printMat("X", X.data(), n, m);
    // printMat("P", P.data(), n, m);

    View1D<T> R_flat(R.data(), n * m);
    residual = KokkosBlas::nrm2(R_flat);

    printf("Iteration %d, residual = %e\n", k, residual);

    k++;
  }

  // vp is the pointer to the eigenvector from X
  View2D<T> v_result(vp, n, m);
  Kokkos::deep_copy(v_result, X);
  // printMat("v", v.data(), n, m);
}

template <typename T>
void lobpcg2(T* Ap, T* Bp, int n, int m, T* wp, T* vp, T* Xp = nullptr,
             T* Mp = nullptr, double tol = 1e-8, int maxiter = 500,
             bool verbose = true) {
  // TODO: Bp is nullptr, use identity matrix
  // TODO: sygv for full matrix eigenvalue problem
  // TODO: if not converged, pick up the result for lowest residual
  // TODO: if the lowest residual is still above 1e-4, then fail
  // TODO: print residual precision based on tolerance set

  const int m0 = m;                    // number of eigenpairs desired
  const int m1 = int(ceil(0.3 * m0));  // added number of eigenpairs to compute
  m = m0 + m1;                         // total number of eigenpairs to compute

  View2D<T> A(Ap, n, n);
  View2D<T> B;
  View2D<T> X("X", n, m);
  View2D<T> M;
  View2D<T> XAX("XAX", m, m);
  View2D<T> XBX("XBX", m, m);

  View2D<T> AX("AX", n, m);
  View2D<T> BX("BX", n, m);

  View1D<T> w("eigenvalues", m);
  View2D<T> v("eigenvectors column major", m, m);

  if (m > int(ceil(n * 0.3))) {
    printf("m is larger than 30%% of m, use Lapack::sygvx instead\n");
    return;
  }

  if (verbose) {
    printf("%d eigenpairs to compute, \n", m0);
    printf("%d eigenpairs added to speed up convergence\n", m1);
  }

  if (Bp == nullptr) {
    B = View2D<T>("B", n, n);
    Kokkos::deep_copy(B, 0.0);
    Kokkos::parallel_for(
        "lobpcg::setB",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, n}),
        KOKKOS_LAMBDA(int i, int j) { B(i, j) = (i == j) ? 1.0 : 0.0; });
  } else {
    B = View2D<T>(Bp, n, n);
  }

  /* Compute XAX = X.T * A * X, XBX = X.T * B * X */
  if (Xp == nullptr) {
    //  X = eye(n, m) -> XAX = A[:m, :m], XBX = B[:m, :m]
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        XAX(i, j) = A(i, j);
        XBX(i, j) = B(i, j);
      }
    }
  } else {
    X = View2D<T>(Xp, n, m);
    KokkosBlas::gemm("N", "N", 1.0, A, X, 0.0, AX);    // AX = A * X
    KokkosBlas::gemm("N", "N", 1.0, B, X, 0.0, BX);    // BX = B * X
    KokkosBlas::gemm("T", "N", 1.0, X, AX, 0.0, XAX);  // XAX = X.T * AX
    KokkosBlas::gemm("T", "N", 1.0, X, BX, 0.0, XBX);  // XBX = X.T * BX
  }

  /* Solve generalized eigenvalue problem to get initial eigenvectors */
  lapackage::sygvx<T>(XAX.data(), XBX.data(), m, m, w.data(), v.data());

  /* Compute: AX = A * X, BX = B * X */
  if (Xp == nullptr) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        X(i, j) = v(j, i);  // X = eye(n, m) -> X = hstack(v, 0)
      }
    }

    auto A_nm = Kokkos::subview(A, Kokkos::ALL(), Kokkos::make_pair(0, m));
    auto B_nm = Kokkos::subview(B, Kokkos::ALL(), Kokkos::make_pair(0, m));
    KokkosBlas::gemm("N", "T", 1.0, A_nm, v, 0.0, AX);  // AX = A * v
    KokkosBlas::gemm("N", "T", 1.0, B_nm, v, 0.0, BX);  // BX = B * v
  } else {
    View2D<T> X0("last X", n, m);
    Kokkos::deep_copy(X0, X);
    KokkosBlas::gemm("N", "T", 1.0, X, v, 0.0, X);   // X = X * v
    KokkosBlas::gemm("N", "N", 1.0, A, X, 0.0, AX);  // AX = A * X
    KokkosBlas::gemm("N", "N", 1.0, B, X, 0.0, BX);  // BX = B * X
  }

  /* Compute residual: R = AX - BX * w, A_norm, B_norm */
  T A_norm = 0.0;
  T B_norm = 0.0;
  T X_norm = 0.0;
  View2D<T> R("residual", n, m);
  for (int i = 0; i < m; i++) {
    auto x = Kokkos::subview(X, Kokkos::ALL(), i);
    auto r = Kokkos::subview(R, Kokkos::ALL(), i);
    auto Ax = Kokkos::subview(AX, Kokkos::ALL(), i);
    auto Bx = Kokkos::subview(BX, Kokkos::ALL(), i);

    A_norm += KokkosBlas::nrm2_squared(Ax);  // A_norm += Ax^2
    B_norm += KokkosBlas::nrm2_squared(Bx);  // B_norm += Bx^2
    X_norm += KokkosBlas::nrm2_squared(x);   // X_norm += x^2

    KokkosBlas::update(1.0, Ax, -w(i), Bx, 0.0, r);  // r = Ax - Bx * w(i)
  }

  A_norm = sqrt(A_norm / X_norm);
  B_norm = sqrt(B_norm / X_norm);

  /* initial convergent array as false: 0 */
  View2D<T> W("W", n, m);
  View2D<T> P("P", n, m);
  View2D<T> S("S", n, 3);
  View2D<T> gramA_in("inner symmetric Gram A matrices", 3, 3);
  View2D<T> gramB_in("inner symmetric Gram B matrices", 3, 3);
  View1D<T> w_in("inner eigenvalues", 1);
  View1D<T> v_in("inner eigenvectors", 3);
  View2D<T> gramA_out("outer symmetric Gram A matrices", m, m);
  View2D<T> gramB_out("outer symmetric Gram B matrices", m, m);
  // View1D<T> w_out("outer eigenvalues", m);
  View2D<T> v_out("outer eigenvectors", m, m);
  View1D<int> is_convergent("convergent flag", m);

  View1D<T> Ax("AX", n);
  View1D<T> Aw("AW", n);
  View1D<T> App("AP", n);
  View1D<T> Bx("BX", n);
  View1D<T> Bw("BW", n);
  View1D<T> Bpp("BP", n);

  /* Start outer loop */
  for (int k = 0; k < maxiter; k++) {
    if (Mp != nullptr) {  // W = M * R, preconditioning, normally M = A^-1
      KokkosBlas::gemm("N", "N", 1.0, M, R, 0.0, W);
    } else {  // without preconditioning
      Kokkos::deep_copy(W, R);
    }

    tick("outer loop: Rayleigh-Ritz procedure 0");
    /* Perform Rayleigh-Ritz procedure */
    /* Z = [Wi, Xi, Pi] */
    View2D<T> Z("Z", n, 3 * m);
    Kokkos::parallel_for(
        "S = [X, W, P]",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, m}),
        KOKKOS_LAMBDA(int i, int j) {
          Z(i, 3 * j) = X(i, j);
          Z(i, 3 * j + 1) = W(i, j);
          Z(i, 3 * j + 2) = P(i, j);
        });

    tock("outer loop: Rayleigh-Ritz procedure 0");

    /* Compute symmetric gram matrices */
    int xm = Z.extent(1);  // number of columns in Z
    View2D<T> gramA("gramA", xm, xm);
    View2D<T> gramB("gramB", xm, xm);
    View2D<T> AS("AS", n, xm);
    View2D<T> BS("BS", n, xm);

    tick("outer loop: Rayleigh-Ritz procedure 1");
    /* gramA = Z^T * A * Z, gramB = Z^T * B * Z */
    KokkosBlas::gemm("N", "N", 1.0, A, Z, 0.0, AS);
    KokkosBlas::gemm("N", "N", 1.0, B, Z, 0.0, BS);
    tock("outer loop: Rayleigh-Ritz procedure 1");

    // /* Perform Rayleigh-Ritz procedure */
    // /* Z = [W, P] */
    // View2D<T> Z("Z", n, 2 * m);
    // Kokkos::parallel_for(
    //     "S = [X, W, P]",
    //     Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, m}),
    //     KOKKOS_LAMBDA(int i, int j) {
    //       Z(i, j) = W(i, j);
    //       Z(i, j + m) = P(i, j);
    //     });

    // /* Compute symmetric gram matrices */

    // tick("outer loop: Rayleigh-Ritz procedure 1");
    // /* gramA = Z^T * A * Z, gramB = Z^T * B * Z */
    // View2D<T> AZ("AS", n, 2 * m);
    // View2D<T> BZ("BS", n, 2 * m);
    // KokkosBlas::gemm("N", "N", 1.0, A, Z, 0.0, AZ);
    // KokkosBlas::gemm("N", "N", 1.0, B, Z, 0.0, BZ);

    // View2D<T> S("Z", n, 3 * m);

    // auto AW = Kokkos::subview(AZ, Kokkos::ALL(), Kokkos::make_pair(0, m));
    // auto BW = Kokkos::subview(BZ, Kokkos::ALL(), Kokkos::make_pair(0, m));
    // auto AP = Kokkos::subview(AZ, Kokkos::ALL(), Kokkos::make_pair(m, 2 *
    // m)); auto BP = Kokkos::subview(BZ, Kokkos::ALL(), Kokkos::make_pair(m, 2
    // * m)); tock("outer loop: Rayleigh-Ritz procedure 1");

    tick("inner loop");
    /* Use hard lock technique to lock the convergent eigenpairs */
    for (int i = 0; i < m; i++) {
      if (is_convergent(i) == 0) {  // only loop over non convergent eigenpairs
        /* Perform inner Rayleigh-Ritz procedure */
        auto As = Kokkos::subview(AS, Kokkos::ALL(),
                                  Kokkos::make_pair(3 * i, 3 * i + 3));
        auto Bs = Kokkos::subview(BS, Kokkos::ALL(),
                                  Kokkos::make_pair(3 * i, 3 * i + 3));
        auto s = Kokkos::subview(Z, Kokkos::ALL(),
                                 Kokkos::make_pair(3 * i, 3 * i + 3));
        KokkosBlas::gemm("T", "N", 1.0, s, As, 0.0, gramA_in);
        KokkosBlas::gemm("T", "N", 1.0, s, Bs, 0.0, gramB_in);

        /* Solve the 3x3 small eigenvalue problem */
        if (k == 0) {
          View2D<T> gramA_sub("gramA_sub", 2, 2);
          View2D<T> gramB_sub("gramB_sub", 2, 2);
          for (int ii = 0; ii < 2; ii++) {
            for (int jj = 0; jj < 2; jj++) {
              gramA_sub(ii, jj) = gramA_in(ii, jj);
              gramB_sub(ii, jj) = gramB_in(ii, jj);
            }
          }

          lapackage::sygvx<T>(gramA_sub.data(), gramB_sub.data(), 2, 1,
                              w_in.data(), v_in.data());
        } else {
          lapackage::sygvx<T>(gramA_in.data(), gramB_in.data(), 3, 1,
                              w_in.data(), v_in.data());
        }
        // tock("inner loop: Rayleigh-Ritz procedure 3");

        // tick("inner loop: Rayleigh-Ritz procedure 4");
        // X[:, i] = X[:, i] * v(0) + W[:, i] * v(1) + P[:, i] * v(2)
        // P[:, i] = W[:, i] * v(1) + P[:, i] * v(2)
        auto x = Kokkos::subview(X, Kokkos::ALL(), i);
        auto w = Kokkos::subview(W, Kokkos::ALL(), i);
        auto p = Kokkos::subview(P, Kokkos::ALL(), i);
        KokkosBlas::axpby(v_in(1), w, v_in(2), p);
        KokkosBlas::axpby(1.0, p, v_in(0), x);

        // tock("inner loop: Rayleigh-Ritz procedure 4")

      }  // end if is_convergent
    }    // end inner loop
    tock("inner loop");

    tick("outer loop: Rayleigh-Ritz procedure 3");
    /* Perform outer Rayleigh-Ritz procedure */
    /*gramA = X^T * A * X, gramB = X^T * B * X */
    KokkosBlas::gemm("N", "N", 1.0, A, X, 0.0, AX);
    KokkosBlas::gemm("N", "N", 1.0, B, X, 0.0, BX);
    KokkosBlas::gemm("T", "N", 1.0, X, AX, 0.0, gramA_out);
    KokkosBlas::gemm("T", "N", 1.0, X, BX, 0.0, gramB_out);
    tock("outer loop: Rayleigh-Ritz procedure 3");

    tick("outer loop: Rayleigh-Ritz procedure 4");
    lapackage::sygvx<T>(gramA_out.data(), gramB_out.data(), m, m, w.data(),
                        v_out.data());
    tock("outer loop: Rayleigh-Ritz procedure 4");

    tick("outer loop: Rayleigh-Ritz procedure 5");
    /* Compute the Ritz vector */
    // View2D<T> X0("X for last iteration", n, m);
    // Kokkos::deep_copy(X0, X);
    // KokkosBlas::gemm("N", "T", 1.0, X0, v_out, 0.0, X);  // X = X * v
    // KokkosBlas::gemm("N", "N", 1.0, A, X, 0.0, AX);      // AX = A * X
    // KokkosBlas::gemm("N", "N", 1.0, B, X, 0.0, BX);      // BX = B * X
    View2D<T> AX0("AX for last iteration", n, m);
    View2D<T> BX0("BX for last iteration", n, m);
    Kokkos::deep_copy(AX0, AX);
    Kokkos::deep_copy(BX0, BX);
    View2D<T> X0("X for last iteration", n, m);
    View2D<T> P0("P for last iteration", n, m);
    Kokkos::deep_copy(X0, X);
    Kokkos::deep_copy(P0, P);
    KokkosBlas::gemm("N", "T", 1.0, X0, v_out, 0.0, X);  // X = X * v
    KokkosBlas::gemm("N", "T", 1.0, P0, v_out, 0.0, P);  // P = P * v
    KokkosBlas::gemm("N", "T", 1.0, AX0, v_out, 0.0, AX);  // AX = AX * v
    KokkosBlas::gemm("N", "T", 1.0, BX0, v_out, 0.0, BX);  // BX = BX * v
    tock("outer loop: Rayleigh-Ritz procedure 5");

    tick("outer loop: Rayleigh-Ritz procedure 6");
    /* Compute residual */
    View1D<T> res("residual norm", m);
    for (int i = 0; i < m; i++) {
      auto x = Kokkos::subview(X, Kokkos::ALL(), i);
      auto Ax = Kokkos::subview(AX, Kokkos::ALL(), i);
      auto Bx = Kokkos::subview(BX, Kokkos::ALL(), i);
      auto r = Kokkos::subview(R, Kokkos::ALL(), i);

      KokkosBlas::update(1.0, Ax, -w(i), Bx, 0.0, r);  // r = Ax - Bx * w(i)
      T x_norm = KokkosBlas::nrm2_squared(x);          // X_norm = ||x||
      T r_norm = KokkosBlas::nrm2_squared(r);          // R_norm = ||r||

      // res = ||ri|| / (||A|| + wi*||B||) * ||xi||)
      res(i) = sqrt(r_norm) / ((A_norm + B_norm * w(i)) * sqrt(x_norm));
    }
    tock("outer loop: Rayleigh-Ritz procedure 6");

    /* Check convergence */
    T res_max = 0.0;
    int count = 0;
    for (int i = 0; i < m0; i++) {
      if (res(i) > res_max) {
        res_max = res(i);
      }
      if (res(i) > tol) {
        is_convergent(i) = 0;  // not converged
      } else {
        is_convergent(i) = 1;  // converged
        count++;
      }
    }

    if (verbose) {
      printf("Iteration %d, %d converged, residual = %e\n", k, count, res_max);
    }

    if (res_max < tol || count == m0) {
      break;
    }

    if (k == maxiter - 1) {
      printf("Warning: maximum number of iterations reached, residual = %e\n",
             res_max);
    }

    for (int i = m0; i < m; i++) {
      if (res(i) > tol) {
        is_convergent(i) = 0;  // not converged
      } else {
        is_convergent(i) = 1;  // converged
      }
    }

  }  // end outer loop

  /* Copy result back to vp */
  View1D<T> w_result(wp, m0);
  View2D<T> v_result(vp, n, m0);
  auto X_m0 = Kokkos::subview(X, Kokkos::ALL(), Kokkos::make_pair(0, m0));
  auto w_m0 = Kokkos::subview(w, Kokkos::make_pair(0, m0));
  Kokkos::deep_copy(v_result, X_m0);
  Kokkos::deep_copy(w_result, w_m0);
}

}  // namespace linalg

#endif  // LOBPCG_HPP

// tick("inner loop: Rayleigh-Ritz procedure 1");

// /* construct S = [X, W, P] */
// Kokkos::parallel_for(
//     "S = [X, W, P]", Kokkos::RangePolicy<ExecSpace>(0, n),
//     KOKKOS_LAMBDA(int j) {
//       S(j, 0) = X(j, i);
//       S(j, 1) = W(j, i);
//       S(j, 2) = P(j, i);
//     });
// View2D<T> AS("AS", n, 3);
// View2D<T> BS("BS", n, 3);
// KokkosBlas::gemm("N", "N", 1.0, A, S, 0.0, AS);
// KokkosBlas::gemm("N", "N", 1.0, B, S, 0.0, BS);

// /* Compute symmetric Gram matrices */
// Kokkos::parallel_for(
//     "Compute: gramA = S^T * A * S, gramB = S^T * B * S",
//     Kokkos::RangePolicy<ExecSpace>(0, n), KOKKOS_LAMBDA(int jj) {
//       gramA_in(0, 0) += X(jj, i) * AS(jj, 0);
//       gramA_in(0, 1) += X(jj, i) * AS(jj, 1);
//       gramA_in(0, 2) += X(jj, i) * AS(jj, 2);
//       gramA_in(1, 1) += W(jj, i) * AS(jj, 1);
//       gramA_in(1, 2) += W(jj, i) * AS(jj, 2);
//       gramA_in(2, 2) += P(jj, i) * AS(jj, 2);

//       gramB_in(0, 0) += X(jj, i) * BS(jj, 0);
//       gramB_in(0, 1) += X(jj, i) * BS(jj, 1);
//       gramB_in(0, 2) += X(jj, i) * BS(jj, 2);
//       gramB_in(1, 1) += W(jj, i) * BS(jj, 1);
//       gramB_in(1, 2) += W(jj, i) * BS(jj, 2);
//       gramB_in(2, 2) += P(jj, i) * BS(jj, 2);
//     });

// // gramA_in(0, 0) = w(i);
// gramA_in(1, 0) = gramA_in(0, 1);
// gramA_in(2, 0) = gramA_in(0, 2);
// gramA_in(2, 1) = gramA_in(1, 2);

// // gramB(0, 0) = 1.0;
// gramB_in(1, 0) = gramB_in(0, 1);
// gramB_in(2, 0) = gramB_in(0, 2);
// gramB_in(2, 1) = gramB_in(1, 2);
// tock("inner loop: Rayleigh-Ritz procedure 1");

// ******************************************************

// tick("inner loop: Rayleigh-Ritz procedure 2");
// /* construct S = [X, W, P] */
// Kokkos::parallel_for(
//     "S = [X, W, P]", Kokkos::RangePolicy<ExecSpace>(0, n),
//     KOKKOS_LAMBDA(int j) {
//       S(j, 0) = X(j, i);
//       S(j, 1) = W(j, i);
//       S(j, 2) = P(j, i);
//     });
// tock("inner loop: Rayleigh-Ritz procedure 2");
// tick("inner loop: Rayleigh-Ritz procedure 2");

// View2D<T> AS("AS", n, 3);
// View2D<T> BS("BS", n, 3);
// KokkosBlas::gemm("N", "N", 1.0, A, S, 0.0, AS);
// KokkosBlas::gemm("N", "N", 1.0, B, S, 0.0, BS);
// tock("inner loop: Rayleigh-Ritz procedure 2");
// tick("inner loop: Rayleigh-Ritz procedure 2");
// KokkosBlas::gemm("T", "N", 1.0, S, AS, 0.0, gramA_in);
// KokkosBlas::gemm("T", "N", 1.0, S, BS, 0.0, gramB_in);
// tock("inner loop: Rayleigh-Ritz procedure 2");

// ****************************************************
// auto Ax = Kokkos::subview(AX, Kokkos::ALL(), i);
// auto Aw = Kokkos::subview(AW, Kokkos::ALL(), i);
// auto App = Kokkos::subview(AP, Kokkos::ALL(), i);

// auto Bx = Kokkos::subview(BX, Kokkos::ALL(), i);
// auto Bw = Kokkos::subview(BW, Kokkos::ALL(), i);
// auto Bpp = Kokkos::subview(BP, Kokkos::ALL(), i);

// auto x = Kokkos::subview(X, Kokkos::ALL(), i);
// auto w = Kokkos::subview(W, Kokkos::ALL(), i);
// auto p = Kokkos::subview(P, Kokkos::ALL(), i);

// gramA_in(0, 0) = KokkosBlas::dot(x, Ax);
// gramA_in(0, 1) = KokkosBlas::dot(x, Aw);
// gramA_in(1, 1) = KokkosBlas::dot(w, Aw);
// gramA_in(0, 2) = KokkosBlas::dot(x, App);
// gramA_in(1, 2) = KokkosBlas::dot(w, App);
// gramA_in(2, 2) = KokkosBlas::dot(p, App);

// gramB_in(0, 0) = 1.0;
// gramB_in(0, 1) = KokkosBlas::dot(x, Bw);
// gramB_in(1, 1) = KokkosBlas::dot(w, Bw);
// gramB_in(0, 2) = KokkosBlas::dot(x, Bpp);
// gramB_in(1, 2) = KokkosBlas::dot(w, Bpp);
// gramB_in(2, 2) = KokkosBlas::dot(p, Bpp);

// gramA_in(1, 0) = gramA_in(0, 1);
// gramA_in(2, 0) = gramA_in(0, 2);
// gramA_in(2, 1) = gramA_in(1, 2);

// gramB_in(1, 0) = gramB_in(0, 1);
// gramB_in(2, 0) = gramB_in(0, 2);
// gramB_in(2, 1) = gramB_in(1, 2);
