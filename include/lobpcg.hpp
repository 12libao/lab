#ifndef LOBPCG_HPP
#define LOBPCG_HPP

#include <KokkosBlas.hpp>
#include <cstdlib>

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
void lobpcg(T* Ap, T* Bp, int n, int m, T* wp, T* vp, T* Xp = nullptr, T* Mp = nullptr,
            double tol = 1e-8, int maxiter = 500, bool verbose = true) {
  // TODO: Bp is nullptr, use identity matrix
  // TODO: sygv for full matrix eigenvalue problem
  // TODO: if not converged, pick up the result for lowest residual
  // TODO: if the lowest residual is still above 1e-4, then fail
  // TODO: print residual precision based on tolerance set

  const int m0 = m;                  // number of eigenpairs desired
  const int m1 = int(ceil(3 * m0));  // added number of eigenpairs to compute
  m = m0 + m1;                       // total number of eigenpairs to compute
  m = n < m ? n : m;                 // m cannot be larger than n

  /* If n < 100, use preconditioned sygvx */
  if (n < 200) {
    m = n; 
  }

  // if (m0 >= int(floor(n * 0.3))) {
  //   printf("\033[1;31mWarning\033[0m: m is larger than 30%% of n.\n");
  //   return;
  // }

  // if (m >= int(floor(n * 0.3))) {
  //   m = int(floor(n * 0.3)) - 1;
  // }

  if (verbose) {
    printf(
        "lobpcg: n = \033[32m%d\033[0m, m = \033[32m%d\033[0m, m (added) = "
        "\033[32m%d\033[0m\n",
        n, m0, m - m0);
  }

  View2D<T> A(Ap, n, n);
  View2D<T> B(Bp, n, n);

  View2D<T> ABXPW("vstack: [X, AX, BX, P, AP, BP, W, AW, BW]", 9 * n, m);
  View2D<T> ABXP0("last iteration: ABXP", 6 * n, m);
  auto ABXP = Kokkos::subview(ABXPW, Kokkos::make_pair(0, 6 * n), Kokkos::ALL());

  auto X_AX_BX = Kokkos::subview(ABXP, Kokkos::make_pair(0 * n, 3 * n), Kokkos::ALL());
  auto P_AP_BP = Kokkos::subview(ABXP, Kokkos::make_pair(3 * n, 6 * n), Kokkos::ALL());
  auto W_AW_BW = Kokkos::subview(ABXP, Kokkos::make_pair(6 * n, 9 * n), Kokkos::ALL());

  auto X = Kokkos::subview(X_AX_BX, Kokkos::make_pair(0, n), Kokkos::ALL());
  auto W = Kokkos::subview(W_AW_BW, Kokkos::make_pair(0, n), Kokkos::ALL());
  auto P = Kokkos::subview(P_AP_BP, Kokkos::make_pair(0, n), Kokkos::ALL());

  auto AX = Kokkos::subview(X_AX_BX, Kokkos::make_pair(n, 2 * n), Kokkos::ALL());
  auto AW = Kokkos::subview(W_AW_BW, Kokkos::make_pair(n, 2 * n), Kokkos::ALL());
  auto AP = Kokkos::subview(P_AP_BP, Kokkos::make_pair(n, 2 * n), Kokkos::ALL());

  auto BX = Kokkos::subview(X_AX_BX, Kokkos::make_pair(2 * n, 3 * n), Kokkos::ALL());
  auto BW = Kokkos::subview(W_AW_BW, Kokkos::make_pair(2 * n, 3 * n), Kokkos::ALL());
  auto BP = Kokkos::subview(P_AP_BP, Kokkos::make_pair(2 * n, 3 * n), Kokkos::ALL());

  /* Compute XAX = X.T * A * X, XBX = X.T * B * X */
  View2D<T> XAX("XAX", m, m);
  View2D<T> XBX("XBX", m, m);
  if (Xp == nullptr) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        XAX(i, j) = A(i, j);  // X = eye(n, m) -> XAX = A[:m, :m]
        XBX(i, j) = B(i, j);  // X = eye(n, m) -> XBX = B[:m, :m]
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
  View1D<T> w("eigenvalues", m);
  View2D<T> v("eigenvectors column major", m, m);
  lapackage::sygvx<T>(XAX.data(), XBX.data(), m, m, w.data(), v.data());

  /* Compute: X = X * v, AX = A * X * v, BX = B * X * v */
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
    KokkosBlas::gemm("N", "N", 1.0, A, X, 0.0, AX);             // AX = A * X
    KokkosBlas::gemm("N", "N", 1.0, B, X, 0.0, BX);             // BX = B * X
    KokkosBlas::gemm("N", "T", 1.0, X_AX_BX, v, 0.0, X_AX_BX);  // X = X * v
  }

  View2D<T> M;
  if (Mp != nullptr) {
    M = View2D<T>(Mp, n, n);
  }

  /* Compute residual: R = AX - BX * w, A_norm, B_norm */
  View1D<T> one("all->1", m);
  View1D<T> neg_one("all->1", m);
  Kokkos::deep_copy(one, 1.0);
  Kokkos::deep_copy(neg_one, -1.0);
  View2D<T> R("R = AX - BX * w", n, m);
  View2D<T> R_("R_ = AX + BX * w", n, m);
  Kokkos::deep_copy(R, AX);              // R = AX
  KokkosBlas::axpby(w, BX, neg_one, R);  // R = R - BX * w

  /* Prepare for outer loop */
  View2D<T> gramAB_outer("outer symmetric Gram A matrices", 2 * m, m);
  auto gramA_outer = Kokkos::subview(gramAB_outer, Kokkos::make_pair(0, m), Kokkos::ALL());
  auto gramB_outer = Kokkos::subview(gramAB_outer, Kokkos::make_pair(m, 2 * m), Kokkos::ALL());

  View2D<T> gramAB_inner("inner symmetric Gram A and B matrices", 6, 3);
  auto gramA_inner = Kokkos::subview(gramAB_inner, Kokkos::make_pair(0, 3), Kokkos::ALL());
  auto gramB_inner = Kokkos::subview(gramAB_inner, Kokkos::make_pair(3, 6), Kokkos::ALL());

  View1D<T> w_outer("outer eigenvalues", m);
  View2D<T> v_outer("outer eigenvectors", m, m);
  View1D<T> w_inner("inner eigenvalues", 1);
  View1D<T> v_inner("inner eigenvectors", 3);

  View1D<T> v0("v0", m);
  View1D<T> v1("v1", m);
  View1D<T> v2("v2", m);
  View2D<T> S("[Xi, Wi, Pi], m sub-blocks[n, 3]", m * n, 3);
  View2D<T> ABS("[AXi, AWi, APi, BXi, BWi, BPi], m sub-blocks[n, 6]", m * n, 6);

  /* initial convergent array as all false: 0 */
  View1D<int> is_convergent("convergent flag", m);

  /* Start outer loop */
  for (int k = 0; k < maxiter; k++) {
    if (Mp != nullptr) {                              // with preconditioning, normally M = A^-1
      KokkosBlas::gemm("N", "N", 1.0, M, R, 0.0, W);  // W = M * R
    } else {                                          // without preconditioning
      Kokkos::deep_copy(W, R);
    }

    if (k == 1) {
      KokkosBlas::gemm("N", "N", 1.0, A, P, 0.0, AP);
      KokkosBlas::gemm("N", "N", 1.0, B, P, 0.0, BP);
    }

    KokkosBlas::gemm("N", "N", 1.0, A, W, 0.0, AW);
    KokkosBlas::gemm("N", "N", 1.0, B, W, 0.0, BW);

    /* Perform Rayleigh-Ritz procedure, ensure stroed in contiguous memory */
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

    /* Perform inner Rayleigh-Ritz procedure */
    /* Use hard lock technique to lock the convergent eigenpairs */
    for (int i = 0; i < m; i++) {
      if (is_convergent(i) == 0) {  // only loop over non convergent eigenpairs

        /* Compute symmetric Gram matrices */
        auto Si = Kokkos::subview(S, Kokkos::make_pair(i * n, i * n + n), Kokkos::ALL());
        auto ABSi = Kokkos::subview(ABS, Kokkos::make_pair(i * n, i * n + n), Kokkos::ALL());
        KokkosBlas::gemm("T", "N", 1.0, ABSi, Si, 0.0, gramAB_inner);

        /* Make sure store gramA, gramB to contigous memory */
        int n_inner = 3;
        if (k == 0) {
          gramA_inner(0, 2) = gramA_inner(1, 0);
          gramA_inner(1, 0) = gramA_inner(1, 1);
          gramB_inner(0, 2) = gramB_inner(1, 0);
          gramB_inner(1, 0) = gramB_inner(1, 1);
          n_inner = 2;
        }

        /* Compute eigenvalues and eigenvectors 3x3 eigenvalue problem */
        lapackage::sygvx<T>(gramA_inner.data(), gramB_inner.data(), n_inner, 1, w_inner.data(),
                            v_inner.data());

        /* Only store the result, move the computation out of the loop */
        v0(i) = v_inner(0);
        v1(i) = v_inner(1);
        v2(i) = v_inner(2);
      }
    }

    /* Compute the Ritz vector, compute batchly except in each column */
    KokkosBlas::axpby(v1, W_AW_BW, v2, P_AP_BP);   // P = W * v(1) + P * v(2)
    KokkosBlas::axpby(one, P_AP_BP, v0, X_AX_BX);  // X = X * v(0) + P

    /* Perform outer Rayleigh-Ritz procedure */
    KokkosBlas::gemm("T", "N", 1.0, X, AX, 0.0, gramA_outer);  // gramA = X^T * AX
    KokkosBlas::gemm("T", "N", 1.0, X, BX, 0.0, gramB_outer);  // gramB = X^T * BX

    /* Compute eigenvalues and eigenvectors for m x m eigenvalue problem */
    lapackage::sygvx<T>(gramA_outer.data(), gramB_outer.data(), m, m, w_outer.data(),
                        v_outer.data());

    /* [X, AX, BX, P, BX, BP] = [X, AX, BX, P, BX, BP] * v */
    Kokkos::deep_copy(ABXP0, ABXP);
    KokkosBlas::gemm("N", "T", 1.0, ABXP0, v_outer, 0.0, ABXP);

    /* Compute Residual: res = ||AX - BX * w|| / ||AX + BX * w|| */
    View1D<T> res("residual norm", m);
    Kokkos::deep_copy(R, AX);                      // R = AX
    KokkosBlas::axpby(w_outer, BX, neg_one, R);    // R = BX * w - R
    KokkosBlas::update(1.0, R, 2.0, AX, 0.0, R_);  // R_ = R + 2*AX

    for (int i = 0; i < m0; i++) {
      if (is_convergent(i) == 0) {
        auto Ri = Kokkos::subview(R, Kokkos::ALL(), i);
        auto Ri_ = Kokkos::subview(R_, Kokkos::ALL(), i);

        T Ri_norm = KokkosBlas::nrm2_squared(Ri);    // ||Ri||^2
        T Ri_norm_ = KokkosBlas::nrm2_squared(Ri_);  // ||Ri_||^2

        res(i) = sqrt(Ri_norm / Ri_norm_);  // res(i) = ||Ri|| / ||Ri_||
      }
    }

    /* Check convergence */
    T res_max = 0.0;
    int count = 0;
    for (int i = 0; i < m0; i++) {
      if (res(i) > res_max) {
        res_max = res(i);  // max residual
      }
      if (res(i) > tol) {
        is_convergent(i) = 0;  // not converged
      } else {
        is_convergent(i) = 1;  // converged
        count++;
      }
    }

    if (verbose) {
      printf(
          "Iteration: \033[32m%2d\033[0m, converged: \033[32m%2d\033[0m, "
          "residual: \033[32m%e\033[0m\n",
          k, count, res_max);
    }

    if (res_max < tol || count == m0) {
      break;
    }

    if (k == maxiter - 1) {
      printf(
          "\033[1;31mWarning\033[0m: maximum number of iterations reached, "
          "residual = %e\n",
          res_max);
    }

  }  // end outer loop

  /* Copy result back to wp, vp */
  View1D<T> w_result(wp, m0);
  View2D<T> v_result(vp, n, m0);
  auto X_m0 = Kokkos::subview(X, Kokkos::ALL(), Kokkos::make_pair(0, m0));
  auto w_m0 = Kokkos::subview(w_outer, Kokkos::make_pair(0, m0));
  Kokkos::deep_copy(v_result, X_m0);
  Kokkos::deep_copy(w_result, w_m0);
}

template <typename T>
void lobpcg_base(T* Ap, T* Bp, int n, int m, T* wp, T* vp, T* Xp = nullptr, T* Mp = nullptr,
                 double tol = 1e-8, int maxiter = 500, bool verbose = true) {
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
        "lobpcg::setB", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, n}),
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
          "S = [X, W, P]", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, m}),
          KOKKOS_LAMBDA(int i, int j) {
            Z(i, j) = W(i, j);
            Z(i, j + m) = X(i, j);
            Z(i, j + 2 * m) = P(i, j);
          });
    } else {
      Z = View2D<T>("Z", n, 2 * m);
      Kokkos::parallel_for(
          "S = [X, W]", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, m}),
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
        "lobpcg::convertEvecs", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {xm, m}),
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
//       gramA_inner(0, 0) += X(jj, i) * AS(jj, 0);
//       gramA_inner(0, 1) += X(jj, i) * AS(jj, 1);
//       gramA_inner(0, 2) += X(jj, i) * AS(jj, 2);
//       gramA_inner(1, 1) += W(jj, i) * AS(jj, 1);
//       gramA_inner(1, 2) += W(jj, i) * AS(jj, 2);
//       gramA_inner(2, 2) += P(jj, i) * AS(jj, 2);

//       gramB_inner(0, 0) += X(jj, i) * BS(jj, 0);
//       gramB_inner(0, 1) += X(jj, i) * BS(jj, 1);
//       gramB_inner(0, 2) += X(jj, i) * BS(jj, 2);
//       gramB_inner(1, 1) += W(jj, i) * BS(jj, 1);
//       gramB_inner(1, 2) += W(jj, i) * BS(jj, 2);
//       gramB_inner(2, 2) += P(jj, i) * BS(jj, 2);
//     });

// // gramA_inner(0, 0) = w(i);
// gramA_inner(1, 0) = gramA_inner(0, 1);
// gramA_inner(2, 0) = gramA_inner(0, 2);
// gramA_inner(2, 1) = gramA_inner(1, 2);

// // gramB(0, 0) = 1.0;
// gramB_inner(1, 0) = gramB_inner(0, 1);
// gramB_inner(2, 0) = gramB_inner(0, 2);
// gramB_inner(2, 1) = gramB_inner(1, 2);
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
// KokkosBlas::gemm("T", "N", 1.0, S, AS, 0.0, gramA_inner);
// KokkosBlas::gemm("T", "N", 1.0, S, BS, 0.0, gramB_inner);
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

// gramA_inner(0, 0) = KokkosBlas::dot(Xi, AXi);
// gramA_inner(0, 1) = KokkosBlas::dot(Xi, AWi);
// gramA_inner(1, 1) = KokkosBlas::dot(Wi, AWi);
// gramA_inner(0, 2) = KokkosBlas::dot(Xi, APi);
// gramA_inner(1, 2) = KokkosBlas::dot(Wi, APi);
// gramA_inner(2, 2) = KokkosBlas::dot(Pi, APi);

// gramB_inner(0, 0) = 1.0;
// gramB_inner(0, 1) = KokkosBlas::dot(Xi, BXi);
// gramB_inner(1, 1) = KokkosBlas::dot(Wi, BWi);
// gramB_inner(0, 2) = KokkosBlas::dot(Xi, BPi);
// gramB_inner(1, 2) = KokkosBlas::dot(Wi, BPi);
// gramB_inner(2, 2) = KokkosBlas::dot(Pi, BPi);

// gramA_inner(1, 0) = gramA_inner(0, 1);
// gramA_inner(2, 0) = gramA_inner(0, 2);
// gramA_inner(2, 1) = gramA_inner(1, 2);

// gramB_inner(1, 0) = gramB_inner(0, 1);
// gramB_inner(2, 0) = gramB_inner(0, 2);
// gramB_inner(2, 1) = gramB_inner(1, 2);