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
    // printmat("W", W.data(), n, m);

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

    // printmat("Z", Z.data(), n, 2 * m);

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

    // printmat("gramA", gramA.data(), 2 * m, 2 * m);
    // printmat("gramB", gramB.data(), 2 * m, 2 * m);

    /* Compute eigenvalues and eigenvectors of reduced eigenvalue problem */
    View1D<T> Ycol("evecs_colmajor", xm * m);
    View2D<T> Y("evecs_rowmajor", xm, m);

    lapackage::sygvx<T>(gramA.data(), gramB.data(), xm, m, wp, Ycol.data());

    // Convert eigenvectors from column-major to row-major
    Kokkos::parallel_for(
        "lobpcg::convertEvecs",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {xm, m}),
        KOKKOS_LAMBDA(int i, int j) { Y(i, j) = Ycol(j * xm + i); });

    // printmat("evals", wp, m, 1);
    // printmat("evecs", Y.data(), xm, m);

    /* Compute Ritz vectors */
    /* Yw = Y[:m, :], Yx = Y[m : 2 * m, :], Yp = Y[2 * m :, :]*/
    auto Yw = Kokkos::subview(Y, Kokkos::make_pair(0, m), Kokkos::ALL());
    auto Yx = Kokkos::subview(Y, Kokkos::make_pair(m, 2 * m), Kokkos::ALL());

    // printmat("Yw", Yw.data(), m, m);
    // printmat("Yx", Yx.data(), m, m);

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

    // printmat("X", X.data(), n, m);
    // printmat("P", P.data(), n, m);

    View1D<T> R_flat(R.data(), n * m);
    residual = KokkosBlas::nrm2(R_flat);

    printf("Iteration %d, residual = %e\n", k, residual);

    k++;
  }

  // vp is the pointer to the eigenvector from X
  View2D<T> v_result(vp, n, m);
  Kokkos::deep_copy(v_result, X);
  // printmat("v", v.data(), n, m);
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

  const int m0 = m;                  // number of eigenpairs desired
  const int m1 = int(ceil(2 * m0));  // added number of eigenpairs to compute
  m = m0 + m1;                       // total number of eigenpairs to compute
  if (m0 >= int(floor(n * 0.3))) {
    printf("\033[31mWarning: m is larger than 30%% of n.\033[0m\n");
    return;
  }

  // if (m >= int(floor(n * 0.3))) {
  //   m = int(floor(n * 0.3)) - 1;
  // }

  if (verbose) {
    printf(
        "lobpcg: n = \033[32m%d\033[0m, m = \033[32m%d\033[0m, m (added) = "
        "\033[32m%d\033[0m\n",
        n, m0, m1);
  }

  // View2D<T> AB("AB", 2 * n, n);
  // auto AB1 = Kokkos::subview(AB, Kokkos::make_pair(0, n), Kokkos::ALL());
  // auto AB2 = Kokkos::subview(AB, Kokkos::make_pair(n, 2 * n), Kokkos::ALL());

  View2D<T> A(Ap, n, n);
  View2D<T> B;
  // View2D<T> X("X", n, m);

  View2D<T> XAX("XAX", m, m);
  View2D<T> XBX("XBX", m, m);

  View2D<T> AS("Z", n, 3 * m);
  View2D<T> BS("Z", n, 3 * m);
  // View2D<T> S("S", n, 3 * m);

  // View2D<T> S("S", n, 3 * m);
  // View2D<T> ASxwp("ASxwp", n, 3 * m);
  // View2D<T> BSxwp("BSxwp", n, 3 * m);
  // View2D<T> gramA_in("inner symmetric Gram A matrices", 3, 3);
  // View2D<T> gramB_in("inner symmetric Gram B matrices", 3, 3);

  // auto pair_3m1 = Kokkos::make_pair(0, 3 * m);
  // auto pair_3m2 = Kokkos::make_pair(3 * m, 6 * m);
  // auto ASxwp = Kokkos::subview(ABS, Kokkos::ALL(), pair_3m1);
  // auto BSxwp = Kokkos::subview(ABS, Kokkos::ALL(), pair_3m2);

  // View2D<T> gramA_sub0("gramA_sub", 2, 2);
  // View2D<T> gramB_sub0("gramB_sub", 2, 2);
  // View2D<T> gramA_sub("gramA_sub", 3, 3);
  // View2D<T> gramB_sub("gramB_sub", 3, 3);

  // auto X = Kokkos::subview(S, Kokkos::ALL(), Kokkos::make_pair(0, m));
  // auto P = Kokkos::subview(S, Kokkos::ALL(), Kokkos::make_pair(m, 2 * m));
  // auto W = Kokkos::subview(S, Kokkos::ALL(), Kokkos::make_pair(2 * m, 3 *
  // m));

  // // auto AX = Kokkos::subview(AS, Kokkos::ALL(), Kokkos::make_pair(0, m));
  // // auto BX = Kokkos::subview(BS, Kokkos::ALL(), Kokkos::make_pair(0, m));
  // // auto AP = Kokkos::subview(AS, Kokkos::ALL(), Kokkos::make_pair(m, 2 *
  // m));
  // // auto BP = Kokkos::subview(BS, Kokkos::ALL(), Kokkos::make_pair(m, 2 *
  // m));
  // // auto AW = Kokkos::subview(AS, Kokkos::ALL(), Kokkos::make_pair(2 * m, 3
  // *
  // // m)); auto BW = Kokkos::subview(BS, Kokkos::ALL(), Kokkos::make_pair(2 *
  // m,
  // // 3 * m));

  // View2D<T> ABX("ABX", 2 * n, m);
  // View2D<T> ABW("ABW", 2 * n, m);
  // View2D<T> ABP("ABP", 2 * n, m);

  // auto AX = Kokkos::subview(ABX, Kokkos::make_pair(0, n), Kokkos::ALL());
  // auto AW = Kokkos::subview(ABW, Kokkos::make_pair(0, n), Kokkos::ALL());
  // auto AP = Kokkos::subview(ABP, Kokkos::make_pair(0, n), Kokkos::ALL());

  // auto BX = Kokkos::subview(ABX, Kokkos::make_pair(n, 2 * n), Kokkos::ALL());
  // auto BW = Kokkos::subview(ABW, Kokkos::make_pair(n, 2 * n), Kokkos::ALL());
  // auto BP = Kokkos::subview(ABP, Kokkos::make_pair(n, 2 * n), Kokkos::ALL());
  View2D<T> ABXP("vstack: [X, AX, BX, P, AP, BP]", 6 * n, m);
  View2D<T> ABXP0("last iteration ABXP", 6 * n, m);
  auto pair_3n1 = Kokkos::make_pair(0, 3 * n);
  auto pair_3n2 = Kokkos::make_pair(3 * n, 6 * n);
  auto X_AX_BX = Kokkos::subview(ABXP, pair_3n1, Kokkos::ALL());
  auto P_AP_BP = Kokkos::subview(ABXP, pair_3n2, Kokkos::ALL());
  // View2D<T> X_AX_BX("vstack[X, AX, BX]", 3 * n, m);
  View2D<T> W_AW_BW("vstack[W, AW, BW]", 3 * n, m);
  // View2D<T> P_AP_BP("vstack[P, AP, BP]", 3 * n, m);

  // View2D<T> P_AP_BP0("P_AP_BP for last iteration", 3 * n, m);

  auto pair_n1 = Kokkos::make_pair(0, n);
  auto X = Kokkos::subview(X_AX_BX, pair_n1, Kokkos::ALL());
  auto W = Kokkos::subview(W_AW_BW, pair_n1, Kokkos::ALL());
  auto P = Kokkos::subview(P_AP_BP, pair_n1, Kokkos::ALL());

  auto pair_n2 = Kokkos::make_pair(n, 2 * n);
  auto AX = Kokkos::subview(X_AX_BX, pair_n2, Kokkos::ALL());
  auto AW = Kokkos::subview(W_AW_BW, pair_n2, Kokkos::ALL());
  auto AP = Kokkos::subview(P_AP_BP, pair_n2, Kokkos::ALL());

  auto pair_n3 = Kokkos::make_pair(2 * n, 3 * n);
  auto BX = Kokkos::subview(X_AX_BX, pair_n3, Kokkos::ALL());
  auto BW = Kokkos::subview(W_AW_BW, pair_n3, Kokkos::ALL());
  auto BP = Kokkos::subview(P_AP_BP, pair_n3, Kokkos::ALL());

  // View2D<T> AX("AX", n, m);
  // View2D<T> BX("BX", n, m);
  // View2D<T> AW("AW", n, m);
  // View2D<T> BW("BW", n, m);
  // View2D<T> AP("AP", n, m);
  // View2D<T> BP("BP", n, m);

  View1D<T> w("eigenvalues", m);
  View2D<T> v("eigenvectors column major", m, m);

  View1D<T> one("all->1", m);
  View1D<T> neg_one("all->1", m);
  Kokkos::deep_copy(one, 1.0);
  Kokkos::deep_copy(neg_one, -1.0);

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
    // KokkosBlas::gemm("N", "T", 1.0, X, v, 0.0, X);   // X = X * v
    KokkosBlas::gemm("N", "N", 1.0, A, X, 0.0, AX);  // AX = A * X
    KokkosBlas::gemm("N", "N", 1.0, B, X, 0.0, BX);  // BX = B * X
    // KokkosBlas::gemm("N", "T", 1.0, AX, v, 0.0, AX);  // AX = A * X
    // KokkosBlas::gemm("N", "T", 1.0, BX, v, 0.0, BX);  // BX = B * X
    KokkosBlas::gemm("N", "T", 1.0, X_AX_BX, v, 0.0, X_AX_BX);  // BX = B * X
  }

  View2D<T> M;
  if (Mp != nullptr) {
    M = View2D<T>(Mp, n, n);
  }

  /* Compute residual: R = AX - BX * w, A_norm, B_norm */
  // T A_norm = 0.0;
  // T B_norm = 0.0;
  // T X_norm = 0.0;
  View2D<T> R("residual", n, m);
  View2D<T> Rd("AX+BX * w", n, m);
  Kokkos::deep_copy(Rd, AX);
  KokkosBlas::axpby(w, BX, neg_one, R);  // R = BX * w
  // KokkosBlas::axpy(-1.0, AX, R);      // R = R - AX
  // for (int i = 0; i < m; i++) {
  //   auto x = Kokkos::subview(X, Kokkos::ALL(), i);
  //   auto r = Kokkos::subview(R, Kokkos::ALL(), i);
  //   auto Ax = Kokkos::subview(AX, Kokkos::ALL(), i);
  //   auto Bx = Kokkos::subview(BX, Kokkos::ALL(), i);

  //   // A_norm += KokkosBlas::nrm2_squared(Ax);  // A_norm += Ax^2
  //   // B_norm += KokkosBlas::nrm2_squared(Bx);  // B_norm += Bx^2
  //   // X_norm += KokkosBlas::nrm2_squared(x);   // X_norm += x^2

  //   KokkosBlas::update(1.0, Ax, -w(i), Bx, 0.0, r);  // r = Ax - Bx * w(i)
  // }

  // A_norm = sqrt(A_norm / X_norm);
  // B_norm = sqrt(B_norm / X_norm);

  /* initial convergent array as false: 0 */
  // View2D<T> W("W", n, m);
  // View2D<T> P("P", n, m);

  // View2D<T> gramA_in("inner symmetric Gram A matrices", 3, 3);
  // View2D<T> gramB_in("inner symmetric Gram B matrices", 3, 3);

  View1D<T> w_in("inner eigenvalues", 1);
  View1D<T> v_in("inner eigenvectors", 3);
  View2D<T> gramAB_out("outer symmetric Gram A matrices", 2 * m, m);
  auto pair_0_m = Kokkos::make_pair(0, m);
  auto pair_m_2m = Kokkos::make_pair(m, 2 * m);

  View2D<T> AXBX("[AX, BX]", n, 2 * m);
  auto AXBX1 = Kokkos::subview(AXBX, Kokkos::ALL(), pair_0_m);
  auto AXBX2 = Kokkos::subview(AXBX, Kokkos::ALL(), pair_m_2m);
  auto gramA_out = Kokkos::subview(gramAB_out, pair_0_m, Kokkos::ALL());
  auto gramB_out = Kokkos::subview(gramAB_out, pair_m_2m, Kokkos::ALL());
  // View2D<T> gramA_out("outer symmetric Gram A matrices", m, m);
  // View2D<T> gramB_out("outer symmetric Gram B matrices", m, m);
  // View1D<T> w_out("outer eigenvalues", m);
  View2D<T> v_out("outer eigenvectors", m, m);
  View1D<int> is_convergent("convergent flag", m);

  // View1D<T> Ax("AX", n);
  // View1D<T> Aw("AW", n);
  // View1D<T> App("AP", n);
  // View1D<T> Bx("BX", n);
  // View1D<T> Bw("BW", n);
  // View1D<T> Bpp("BP", n);

  // View2D<T>
  // printmat("X", X.data(), n, m);
  // printmat("W", R.data(), n, m);
  // printmat("P", P.data(), n, m);

  // View2D<T> Si("Si", n, 3* m);
  // View2D<T> ASi("ASi", n, 3*m);
  // View2D<T> BSi("BSi", n, 3*m);
  // tick("copying");
  // Kokkos::deep_copy(AB1, A);
  // Kokkos::deep_copy(AB2, B);
  // // Kokkos::parallel_for(
  // //     "copying",
  // //     Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, n}),
  // //     KOKKOS_LAMBDA(int i, int j) {
  // //       AB1(i, j) = A(i, j);
  // //       AB2(i, j) = B(i, j);
  // //     });
  // tock("copying");

  View1D<T> v0("v0", m);
  View1D<T> v1("v1", m);
  View1D<T> v2("v2", m);

  // Kokkos::LayoutStride layout_n_1_3m_m(n, 1, 3 * m, m);
  // Kokkos::View<T*, Kokkos::LayoutStride> S("[Xi, Wi, Pi], m sub-blocks");
  // S = Kokkos::View<T*, Kokkos::LayoutStride>("S", layout_n_1_3m_m);
  // View2D<T> ABS("[AXi, AWi, APi, BXi, BWi, BPi], m sub-blocks", n, 6 * m);
  // Kokkos::LayoutStride layout_n_1_3m_m(n, 1, 3 * m, m);
  View2D<T> S("[Xi, Wi, Pi], m sub-blocks[n, 3]", m * n, 3);
  View2D<T> ABS("[AXi, AWi, APi, BXi, BWi, BPi], m sub-blocks[n, 6]", m * n, 6);

  // View2D<T> gramAB_in("inner symmetric Gram A and B matrices", 3, 6);
  // auto pair_03 = Kokkos::make_pair(0, 3);
  // auto pair_36 = Kokkos::make_pair(3, 6);
  // auto gramA_in = Kokkos::subview(gramAB_in, Kokkos::ALL(), pair_03);
  // auto gramB_in = Kokkos::subview(gramAB_in, Kokkos::ALL(), pair_36);
  View2D<T> gramAB_in("inner symmetric Gram A and B matrices", 6, 3);
  auto pair_0_3 = Kokkos::make_pair(0, 3);
  auto pair_3_6 = Kokkos::make_pair(3, 6);
  auto gramA_in = Kokkos::subview(gramAB_in, pair_0_3, Kokkos::ALL());
  auto gramB_in = Kokkos::subview(gramAB_in, pair_3_6, Kokkos::ALL());

  /* Start outer loop */
  for (int k = 0; k < maxiter; k++) {
    tick("outer loop: Rayleigh-Ritz procedure 0");
    if (Mp != nullptr) {  // W = M * R, preconditioning, normally M = A^-1
      KokkosBlas::gemm("N", "N", 1.0, M, R, 0.0, W);
    } else {  // without preconditioning
      Kokkos::deep_copy(W, R);
    }

    if (k == 1) {
      // View2D<T> ABP("ABP", 2 * n, m);
      // KokkosBlas::gemm("N", "N", 1.0, AB, P, 0.0, ABP);
      // auto AP = Kokkos::subview(ABP, Kokkos::make_pair(0, n),  Kokkos::ALL()
      // ); auto BP = Kokkos::subview(ABP, Kokkos::make_pair(n, 2*n),
      // Kokkos::ALL() );

      KokkosBlas::gemm("N", "N", 1.0, A, P, 0.0, AP);
      KokkosBlas::gemm("N", "N", 1.0, B, P, 0.0, BP);
    }

    KokkosBlas::gemm("N", "N", 1.0, A, W, 0.0, AW);
    KokkosBlas::gemm("N", "N", 1.0, B, W, 0.0, BW);
    tock("outer loop: Rayleigh-Ritz procedure 0");
    // printmat("AW", AW.data(), n, m);
    // printmat("BW", BW.data(), n, m);

    // View2D<T> ABW("ABW", 2 * n, m);
    // KokkosBlas::gemm("N", "N", 1.0, AB, W, 0.0, ABW);
    // // printmat("ABW", ABW.data(), 2*n, m);
    // AW = Kokkos::subview(ABW, Kokkos::make_pair(0, n),  Kokkos::ALL() );
    // BW = Kokkos::subview(ABW, Kokkos::make_pair(n, 2*n),  Kokkos::ALL() );

    // printmat("AW", AW.data(), n, m);
    // printmat("BW", BW.data(), n, m);

    // tick("outer loop: Rayleigh-Ritz procedure 1");
    // /* Perform Rayleigh-Ritz procedure */
    // /* Z = [Wi, Xi, Pi] */

    // Kokkos::parallel_for(
    //     "S = [X, W, P]",
    //     Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, m}),
    //     KOKKOS_LAMBDA(int i, int j) {
    //       AS(i, 3 * j) = AX(i, j);
    //       AS(i, 3 * j + 1) = AW(i, j);
    //       AS(i, 3 * j + 2) = AP(i, j);
    //       BS(i, 3 * j) = BX(i, j);
    //       BS(i, 3 * j + 1) = BW(i, j);
    //       BS(i, 3 * j + 2) = BP(i, j);
    //       S(i, 3 * j) = X(i, j);
    //       S(i, 3 * j + 1) = W(i, j);
    //       S(i, 3 * j + 2) = P(i, j);
    //     });
    // tock("outer loop: Rayleigh-Ritz procedure 1");

    // printmat("S", S.data(), n, 3 * m);

    /* Compute symmetric gram matrices */
    // int xm = Z.extent(1);  // number of columns in Z
    // View2D<T> gramA("gramA", xm, xm);
    // View2D<T> gramB("gramB", xm, xm);
    // View2D<T> AS("AS", n, xm);
    // View2D<T> BS("BS", n, xm);

    // tick("outer loop: Rayleigh-Ritz procedure 1");
    /* gramA = Z^T * A * Z, gramB = Z^T * B * Z */
    // KokkosBlas::gemm("N", "N", 1.0, A, Z, 0.0, AS);
    // KokkosBlas::gemm("N", "N", 1.0, B, Z, 0.0, BS);
    // tock("outer loop: Rayleigh-Ritz procedure 1");

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

    tick("outer loop: Rayleigh-Ritz procedure 8");
    Kokkos::parallel_for(
        "S = [Xi, Wi, Pi], m sub-blocks[n, 3] in vstack"
        "ABS = [AXi, AWi, APi, BXi, BWi, BPi], m sub-blocks[n, 6] in vstack",
        Kokkos::RangePolicy<ExecSpace>(0, n), KOKKOS_LAMBDA(const int i) {
          for (int j = 0; j < m; ++j) {
            if (is_convergent(j) == 0) {
              S(i + n * j, 0) = X(i, j);
              S(i + n * j, 1) = W(i, j);
              S(i + n * j, 2) = P(i, j);

              ABS(i + n * j, 0) = AX(i, j);
              ABS(i + n * j, 1) = AW(i, j);
              ABS(i + n * j, 2) = AP(i, j);
              ABS(i + n * j, 3) = BX(i, j);
              ABS(i + n * j, 4) = BW(i, j);
              ABS(i + n * j, 5) = BP(i, j);
            }
          }
        });

    // Wait for all the threads to finish before proceeding.
    // Kokkos::fence();
    tock("outer loop: Rayleigh-Ritz procedure 8");

    tick("inner loop");
    /* Use hard lock technique to lock the convergent eigenpairs */
    for (int i = 0; i < m; i++) {
      if (is_convergent(i) == 0) {  // only loop over non convergent eigenpairs
        /* Perform inner Rayleigh-Ritz procedure */

        tick("inner loop: Rayleigh-Ritz procedure 1");
        /* Si = [Xi, Wi, Pi] */

        /* Compute symmetric Gram matrices */

        // auto pair = Kokkos::make_pair(3 * i, 3 * i + 3);
        auto pair_n = Kokkos::make_pair(i * n, i * n + n);
        auto Si = Kokkos::subview(S, pair_n, Kokkos::ALL());
        // auto ASi = Kokkos::subview(ASxwp, Kokkos::ALL(), pair);
        // auto BSi = Kokkos::subview(BSxwp, Kokkos::ALL(), pair);
        auto ABSi = Kokkos::subview(ABS, pair_n, Kokkos::ALL());

        // View2D<T> Si1("ASi", n, 3);
        // View2D<T> ABSi1("BSi", n, 6);
        // Kokkos::deep_copy(Si1, Si);
        // Kokkos::deep_copy(ABSi1, ABSi);
        // Kokkos::parallel_for(
        //     "Si = [Xi, Wi, Pi]",
        //     Kokkos::RangePolicy<ExecSpace>(0, n), KOKKOS_LAMBDA(int ii) {
        //       Si(ii, 0) = X(ii, i);
        //       Si(ii, 1) = W(ii, i);
        //       Si(ii, 2) = P(ii, i);

        //       ABSi(ii, 0) = AX(ii, i);
        //       ABSi(ii, 1) = AW(ii, i);
        //       ABSi(ii, 2) = AP(ii, i);
        //       ABSi(ii, 3) = BX(ii, i);
        //       ABSi(ii, 4) = BW(ii, i);
        //       ABSi(ii, 5) = BP(ii, i);
        //     });

        // if(i == 0){
        //   printmat("ASi", ASi.data(), n, 3);
        //   printmat("BSi", BSi.data(), n, 3);
        //   printmat("ABSi", ABSi.data(), n, 6);
        // }

        // printmat("Si", Si.data(), n, 3);
        // printmat("ASi", ASi.data(), n, 3);
        // printmat("BSi", BSi.data(), n, 3);

        tock("inner loop: Rayleigh-Ritz procedure 1");

        // auto ASi = Kokkos::subview(AS, Kokkos::ALL(),
        //                           Kokkos::make_pair(

        // printmat("Si", Si.data(), n, 3);
        // printmat("ASi", ASi.data(), n, 3);
        // printmat("BSi", BSi.data(), n, 3);
        tick("inner loop: Rayleigh-Ritz procedure 2");
        KokkosBlas::gemm("T", "N", 1.0, ABSi, Si, 0.0, gramAB_in);

        // KokkosBlas::gemm("T", "N", 1.0, Si, ASi, 0.0, gramA_in);
        // KokkosBlas::gemm("T", "N", 1.0, Si, BSi, 0.0, gramB_in);

        tock("inner loop: Rayleigh-Ritz procedure 2");
        // printmat("gramA_in", gramA_in.data(), 3, 3);
        // printmat("gramB_in", gramB_in.data(), 3, 3);

        tick("inner loop: Rayleigh-Ritz procedure 3");
        /* Solve the 3x3 small eigenvalue problem */
        // int n_in = 3;
        int n_in = 3;
        if (k == 0) {  // copy to contigous memory
          gramA_in(0, 2) = gramA_in(1, 0);
          gramA_in(1, 0) = gramA_in(1, 1);
          gramB_in(0, 2) = gramB_in(1, 0);
          gramB_in(1, 0) = gramB_in(1, 1);
          n_in = 2;
        }
        // View2D<T> gramA_sub("gramA_sub", n_in, n_in);
        // View2D<T> gramB_sub("gramB_sub", n_in, n_in);
        // for (int ii = 0; ii < n_in; ii++) {
        //   for (int jj = 0; jj < n_in; jj++) {
        //     gramA_sub(ii, jj) = gramA_in(ii, jj);  // copy to contigous
        //     memory gramB_sub(ii, jj) = gramB_in(ii, jj);  // copy to
        //     contigous memory
        //   }
        // }
        tock("inner loop: Rayleigh-Ritz procedure 3");

        tick("inner loop: Rayleigh-Ritz procedure 4");

        lapackage::sygvx<T>(gramA_in.data(), gramB_in.data(), n_in, 1,
                            w_in.data(), v_in.data());

        // printmat("w_in", w_in.data(), 1, 1);

        // Xi = Xi * v(0) + Wi * v(1) + Pi * v(2)
        // Pi = Wi * v(1) + Pi * v(2)
        // auto Xi = Kokkos::subview(X, Kokkos::ALL(), i);
        // auto Wi = Kokkos::subview(W, Kokkos::ALL(), i);
        // auto Pi = Kokkos::subview(P, Kokkos::ALL(), i);
        // KokkosBlas::axpby(v_in(1), Wi, v_in(2), Pi);
        // KokkosBlas::axpby(1.0, Pi, v_in(0), Xi);

        // auto ABXi = Kokkos::subview(ABX, Kokkos::ALL(), i);
        // auto ABWi = Kokkos::subview(ABW, Kokkos::ALL(), i);
        // auto ABPi = Kokkos::subview(ABP, Kokkos::ALL(), i);
        // KokkosBlas::axpby(v_in(1), ABWi, v_in(2), ABPi);
        // KokkosBlas::axpby(1.0, ABPi, v_in(0), ABXi);

        // auto X_AX_BX_i = Kokkos::subview(X_AX_BX, Kokkos::ALL(), i);
        // auto W_AW_BW_i = Kokkos::subview(W_AW_BW, Kokkos::ALL(), i);
        // auto P_AP_BP_i = Kokkos::subview(P_AP_BP, Kokkos::ALL(), i);
        // KokkosBlas::axpby(v_in(1), W_AW_BW_i, v_in(2), P_AP_BP_i);
        // KokkosBlas::axpby(1.0, P_AP_BP_i, v_in(0), X_AX_BX_i);
        v0(i) = v_in(0);
        v1(i) = v_in(1);
        v2(i) = v_in(2);

        tock("inner loop: Rayleigh-Ritz procedure 4")

      }  // end if is_convergent
    }    // end inner loop
    tock("inner loop");

    tick("outer loop: Rayleigh-Ritz procedure 9");
    /* update X, P, AX, AP, BX, BP, compute batchly except in each column */
    KokkosBlas::axpby(v1, W_AW_BW, v2, P_AP_BP);   // P = W * v(1) + P * v(2)
    KokkosBlas::axpby(one, P_AP_BP, v0, X_AX_BX);  // X = P + X * v(0)

    tock("outer loop: Rayleigh-Ritz procedure 9");

    tick("outer loop: Rayleigh-Ritz procedure 3");
    /* Perform outer Rayleigh-Ritz procedure */
    /*gramA = X^T * A * X, gramB = X^T * B * X */
    // KokkosBlas::gemm("N", "N", 1.0, A, X, 0.0, AX);
    // KokkosBlas::gemm("N", "N", 1.0, B, X, 0.0, BX);

    // Kokkos::deep_copy(AXBX1, AX);
    // Kokkos::deep_copy(AXBX2, BX);
    // KokkosBlas::gemm("T", "N", 1.0, AXBX, X, 0.0, gramAB_out);
    KokkosBlas::gemm("T", "N", 1.0, X, AX, 0.0, gramA_out);
    KokkosBlas::gemm("T", "N", 1.0, X, BX, 0.0, gramB_out);
    tock("outer loop: Rayleigh-Ritz procedure 3");

    tick("outer loop: Rayleigh-Ritz procedure 4");
    lapackage::sygvx<T>(gramA_out.data(), gramB_out.data(), m, m, w.data(),
                        v_out.data());
    tock("outer loop: Rayleigh-Ritz procedure 4");

    tick("outer loop: Rayleigh-Ritz procedure 5");
    /* Compute the Ritz vector */
    /* [X, AX, BX, P, BX, BP] = [X, AX, BX, P, BX, BP] * v */
    Kokkos::deep_copy(ABXP0, ABXP);
    KokkosBlas::gemm("N", "T", 1.0, ABXP0, v_out, 0.0, ABXP);
    // Kokkos::deep_copy(X_AX_BX0, X_AX_BX);
    // Kokkos::deep_copy(P_AP_BP0, P_AP_BP);
    // KokkosBlas::gemm("N", "T", 1.0, X_AX_BX0, v_out, 0.0, X_AX_BX);
    // KokkosBlas::gemm("N", "T", 1.0, P_AP_BP0, v_out, 0.0, P_AP_BP);

    tock("outer loop: Rayleigh-Ritz procedure 5");

    tick("outer loop: Rayleigh-Ritz procedure 6");
    /* Compute residual */
    Kokkos::deep_copy(R, AX);              // R = AX
    KokkosBlas::axpby(w, BX, neg_one, R);  // R = BX * w - R
    // KokkosBlas::axpy(-1.0, AX, R);  // R = R - AX
    KokkosBlas::update(1.0, R, 2.0, AX, 0.0, Rd);  // Rd = R + 2*AX

    tock("outer loop: Rayleigh-Ritz procedure 6");

    tick("outer loop: Rayleigh-Ritz procedure 7");
    View1D<T> res("residual norm", m);
    for (int i = 0; i < m0; i++) {
      if (is_convergent(i) == 0) {
        // auto Xi = Kokkos::subview(X, Kokkos::ALL(), i);
        // auto AXi = Kokkos::subview(AX, Kokkos::ALL(), i);
        // auto BXi = Kokkos::subview(BX, Kokkos::ALL(), i);
        auto Ri = Kokkos::subview(R, Kokkos::ALL(), i);
        auto Rdi = Kokkos::subview(Rd, Kokkos::ALL(), i);

        // res(i) = sqrt(KokkosBlas::nrm2w_squared(Ri, Rdi));  // res = ||Ri|| /
        // ||Rdi||

        // res = ||ri|| / (||A|| + wi*||B||) * ||xi||)
        // T Xi_norm = KokkosBlas::nrm2_squared(Xi);
        // T AXi_norm = sqrt(KokkosBlas::nrm2_squared(AXi));
        // T BXi_norm = sqrt(KokkosBlas::nrm2_squared(BXi));
        T Ri_norm = KokkosBlas::nrm2_squared(Ri);
        T Rdi_norm = KokkosBlas::nrm2_squared(Rdi);
        // res(i) = sqrt(Ri_norm) / ((A_norm + B_norm * w(i)) * sqrt(Xi_norm));
        // res(i) = Ri_norm / (AXi_norm + BXi_norm * w(i));
        res(i) = sqrt(Ri_norm / Rdi_norm);
        // res(i) = Ri_norm;
      }
    }

    tock("outer loop: Rayleigh-Ritz procedure 7");

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
      // printf("Iteration %d, %d converged, residual = %e\n", k, count,
      // res_max);
      printf("Iteration \033[1;31m%d\033[0m, %d converged, residual = %e\n", k,
             count, res_max);
    }

    if (res_max < tol || count == m0) {
      break;
    }

    if (k == maxiter - 1) {
      // printf("Warning: maximum number of iterations reached, residual =
      // %e\n",
      //        res_max);
      printf(
          "\033[1;31mWarning\033[0m: maximum number of iterations reached, "
          "residual = %e\n",
          res_max);
    }

    // for (int i = m0; i < m; i++) {
    //   if (res(i) > tol) {
    //     is_convergent(i) = 0;  // not converged
    //   } else {
    //     is_convergent(i) = 1;  // converged
    //   }
    // }

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
// auto AXi = Kokkos::subview(AX, Kokkos::ALL(), i);
// auto AWi = Kokkos::subview(AW, Kokkos::ALL(), i);
// auto APi = Kokkos::subview(AP, Kokkos::ALL(), i);

// auto BXi = Kokkos::subview(BX, Kokkos::ALL(), i);
// auto BWi = Kokkos::subview(BW, Kokkos::ALL(), i);
// auto BPi = Kokkos::subview(BP, Kokkos::ALL(), i);

// auto Xi = Kokkos::subview(X, Kokkos::ALL(), i);
// auto Wi = Kokkos::subview(W, Kokkos::ALL(), i);
// auto Pi = Kokkos::subview(P, Kokkos::ALL(), i);

// gramA_in(0, 0) = KokkosBlas::dot(Xi, AXi);
// gramA_in(0, 1) = KokkosBlas::dot(Xi, AWi);
// gramA_in(1, 1) = KokkosBlas::dot(Wi, AWi);
// gramA_in(0, 2) = KokkosBlas::dot(Xi, APi);
// gramA_in(1, 2) = KokkosBlas::dot(Wi, APi);
// gramA_in(2, 2) = KokkosBlas::dot(Pi, APi);

// gramB_in(0, 0) = 1.0;
// gramB_in(0, 1) = KokkosBlas::dot(Xi, BXi);
// gramB_in(1, 1) = KokkosBlas::dot(Wi, BWi);
// gramB_in(0, 2) = KokkosBlas::dot(Xi, BPi);
// gramB_in(1, 2) = KokkosBlas::dot(Wi, BPi);
// gramB_in(2, 2) = KokkosBlas::dot(Pi, BPi);

// gramA_in(1, 0) = gramA_in(0, 1);
// gramA_in(2, 0) = gramA_in(0, 2);
// gramA_in(2, 1) = gramA_in(1, 2);

// gramB_in(1, 0) = gramB_in(0, 1);
// gramB_in(2, 0) = gramB_in(0, 2);
// gramB_in(2, 1) = gramB_in(1, 2);