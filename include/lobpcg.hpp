#pragma once
#ifndef LOBPCG_HPP
#define LOBPCG_HPP

#include <KokkosBlas.hpp>
#include <Kokkos_DualView.hpp>
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
template <typename T>
using HostView1D = Kokkos::View<T*, Kokkos::LayoutLeft, Kokkos::HostSpace>;
template <typename T>
using HostView2D = Kokkos::View<T**, Kokkos::LayoutLeft, Kokkos::HostSpace>;

namespace linalg {
/**
 * Given a real symmetric 3x3 matrix A, compute selected eigenvalues and eigenvectors use analytical
 * method
 *
 * Input:
 *   Ap: real symmetric 3x3 matrix A
 *    m: first m eigenvalues and eigenvectors to be computed
 *
 * Output:
 *   wp: first m eigenvalues of A
 *   vp: first m eigenvectors of A
 *
 * Note:
 *    1. use analytical method
 *    2. eigenvalues and eigenvectors are sorted in ascending order
 *    3. 'A' have to be normalised, otherwise the result will lose precision for case ¦Ëi >> ¦Ëj
 *
 * Reference:
 *    1. https://en.wikipedia.org/wiki/Eigenvalue_algorithm#cite_note-Smith-19
 *    2. Smith, Oliver K. (April 1961), "Eigenvalues of a symmetric 3 ¡Á 3 matrix.", Communications
 * of the ACM, 4 (4): 168, doi:10.1145/355578.366316, S2CID 37815415
 */
template <typename T>
void syevx3x3_analytical(T* Ap, int m, T* wp, T* vp, bool verbose = true) {
  constexpr auto pi = Kokkos::numbers::pi_v<T>;
  View2D<T> A(Ap, 3, 3);  // RowMajor or ColMajor are the same since A is symmetric
  View1D<T> w(wp, m);
  View2D<T> v(vp, 3, m);  // RowMajor for host, ColMajor for device

  T a00 = A(0, 0);
  T a01 = A(0, 1);
  T a02 = A(0, 2);
  T a11 = A(1, 1);
  T a12 = A(1, 2);
  T a22 = A(2, 2);

  T p1 = a01 * a01 + a02 * a02 + a12 * a12;

  /* Check if matrix is diagonal */
  if (p1 == 0) {
    for (int i = 0; i < m; ++i) {
      w(i) = A(i, i);  // eigenvalues are diagonal elements
      for (int j = 0; j < m; ++j) {
        v(j, i) = (i == j) ? 1 : 0;  // eigenvectors are the identity matrix
      }
    }
    return;
  }

  T q = (a00 + a11 + a22) / 3;  // trace(A) / 3

  T b00 = a00 - q;
  T b11 = a11 - q;
  T b22 = a22 - q;

  T p = sqrt((b00 * b00 + b11 * b11 + b22 * b22 + 2 * p1) / 6);  // norm(A - q * I) / sqrt(6)

  /* Compute the determinant of B */
  T detB = (b00 * (b11 * b22 - a12 * a12) - a01 * (a01 * b22 - a12 * a02) +
            a02 * (a01 * a12 - b11 * a02));

  T r = detB / (2 * p * p * p);

  // In exact arithmetic for a symmetric matrix  -1 <= r <= 1
  // but computation error can leave it slightly outside this range.
  T phi;
  if (r <= -1)
    phi = pi / 3;
  else if (r >= 1)
    phi = 0;
  else
    phi = Kokkos::acos(r) / 3;

  /* Compute eigenvalues, the eigenvalues satisfy ¦Ë0 <= ¦Ë1 <= ¦Ë2 */
  T w0 = q + 2 * p * Kokkos::cos(phi + (2 * pi / 3));
  T w1 = q + 2 * p * Kokkos::cos(phi - (2 * pi / 3));
  T w2 = 3 * q - w0 - w1;  // since trace(A) = eig1 + eig2 + eig3

  /* Compute eigenvectors */
  /* v[:, 0] = (A - w(1) * I) * (A - w(2) * I)[: , 1] */
  v(0, 0) = (a00 - w1) * a01 + a01 * (a11 - w2) + a02 * a12;
  v(1, 0) = a01 * a01 + (a11 - w1) * (a11 - w2) + a12 * a12;
  v(2, 0) = a02 * a01 + a12 * (a11 - w2) + (a22 - w1) * a12;

  T norm1 = sqrt(v(0, 0) * v(0, 0) + v(1, 0) * v(1, 0) + v(2, 0) * v(2, 0));

  w(0) = w0;
  v(0, 0) /= norm1;
  v(1, 0) /= norm1;
  v(2, 0) /= norm1;

  /* v[:, 1] = (A - w(2) * I) * (A - w(0) * I)[: , 2] */
  if (m > 1) {
    v(0, 1) = (a00 - w2) * a02 + a01 * a12 + a02 * (a22 - w0);
    v(1, 1) = a01 * a02 + (a11 - w2) * a12 + a12 * (a22 - w0);
    v(2, 1) = a02 * a02 + a12 * a12 + (a22 - w2) * (a22 - w0);

    T norm2 = sqrt(v(0, 1) * v(0, 1) + v(1, 1) * v(1, 1) + v(2, 1) * v(2, 1));

    w(1) = w1;
    v(0, 1) /= norm2;
    v(1, 1) /= norm2;
    v(2, 1) /= norm2;
  }

  /* v[:, 2] = (A - w(0) * I) * (A - w(1) * I)[: , 0] */
  if (m > 2) {
    v(0, 2) = (a00 - w0) * (a00 - w1) + a01 * a01 + a02 * a02;
    v(1, 2) = a01 * (a00 - w1) + (a11 - w0) * a01 + a12 * a02;
    v(2, 2) = a02 * (a00 - w1) + a12 * a01 + (a22 - w0) * a02;

    T norm3 = sqrt(v(0, 2) * v(0, 2) + v(1, 2) * v(1, 2) + v(2, 2) * v(2, 2));

    w(2) = w2;
    v(0, 2) /= norm3;
    v(1, 2) /= norm3;
    v(2, 2) /= norm3;
  }
}

/**
 * Computes selected eigenpairs for 3x3 real generalized symmetric-definite eigenproblem Ax=¦ËBx
 *
 * Input:
 *   Ap: pointer for real symmetric 3x3 matrix A
 *   Bp: pointer for real symmetric 3x3 matrix B
 *    m: first m eigenvalues and eigenvectors to be computed
 *
 * Output:
 *   wp: first m eigenvalues of A
 *   vp: first m eigenvectors of A
 *
 * Note:
 *    1. Algorithm 1 in reference 1
 *    2. eigenvalues and eigenvectors are sorted in ascending order
 *    3. 'A' have to be normalised, otherwise the result will lose precision for case ¦Ëi >> ¦Ëj
 *
 * Algorithm:
 *    1. ¦µB, ¦«B <- B * ¦µB = ¦µB * ¦«B
 *    2. ¦µB_hat <- ¦µB_hat = ¦µB * ¦«B^(?1/2) ¡Ö ¦µB * (¦«B^(1/2) + ¦ÅI)^(?1)
 *    3. A_hat <- A_hat = ¦µB_hat * A * ¦µB_hat
 *    4. ¦µA, ¦«A <- A_hat * ¦µA = ¦µA * ¦«A
 *    5. ¦« <- ¦«A, ¦µ <- ¦µB_hat * ¦µA
 *
 * Reference:
 *    1. Ghojogh B, Karray F, Crowley M. Eigenvalue and generalized eigenvalue problems:
 * Tutorial[J]. arXiv preprint arXiv:1903.11240, 2019.
 */
template <typename T>
void sygvx3x3(T* Ap, T* Bp, int m, T* wp, T* vp, bool verbose = true) {
  View2D<T> A(Ap, 3, 3);
  View2D<T> B(Bp, 3, 3);

  /* Compute eigenvalues and eigenvectors of B */
  View2D<T> vB("eigenvectors of B", 3, 3);
  View1D<T> wB("eigenvalues of B", 3);
  syevx3x3_analytical(Bp, 3, wB.data(), vB.data());

  /* Compute ¦µB_hat = ¦µB * (¦«B^(1/2) + ¦ÅI)^(?1), in case ¦«B^(1/2) is singular */
  T eps = std::numeric_limits<T>::epsilon();
  wB(0) = 1 / (sqrt(wB(0)) + eps);
  wB(1) = 1 / (sqrt(wB(1)) + eps);
  wB(2) = 1 / (sqrt(wB(2)) + eps);

  vB(0, 0) *= wB(0);
  vB(1, 0) *= wB(0);
  vB(2, 0) *= wB(0);

  vB(0, 1) *= wB(1);
  vB(1, 1) *= wB(1);
  vB(2, 1) *= wB(1);

  vB(0, 2) *= wB(2);
  vB(1, 2) *= wB(2);
  vB(2, 2) *= wB(2);

  /* Compute A_hat = ¦µB_hat * A * ¦µB_hat */
  View2D<T> A_hat("A_hat", 3, 3);
  T a00 = A(0, 0) * vB(0, 0) + A(0, 1) * vB(1, 0) + A(0, 2) * vB(2, 0);
  T a10 = A(1, 0) * vB(0, 0) + A(1, 1) * vB(1, 0) + A(1, 2) * vB(2, 0);
  T a20 = A(2, 0) * vB(0, 0) + A(2, 1) * vB(1, 0) + A(2, 2) * vB(2, 0);

  T a01 = A(0, 0) * vB(0, 1) + A(0, 1) * vB(1, 1) + A(0, 2) * vB(2, 1);
  T a11 = A(1, 0) * vB(0, 1) + A(1, 1) * vB(1, 1) + A(1, 2) * vB(2, 1);
  T a21 = A(2, 0) * vB(0, 1) + A(2, 1) * vB(1, 1) + A(2, 2) * vB(2, 1);

  T a02 = A(0, 0) * vB(0, 2) + A(0, 1) * vB(1, 2) + A(0, 2) * vB(2, 2);
  T a12 = A(1, 0) * vB(0, 2) + A(1, 1) * vB(1, 2) + A(1, 2) * vB(2, 2);
  T a22 = A(2, 0) * vB(0, 2) + A(2, 1) * vB(1, 2) + A(2, 2) * vB(2, 2);

  A_hat(0, 0) = vB(0, 0) * a00 + vB(1, 0) * a10 + vB(2, 0) * a20;
  A_hat(0, 1) = vB(0, 0) * a01 + vB(1, 0) * a11 + vB(2, 0) * a21;
  A_hat(0, 2) = vB(0, 0) * a02 + vB(1, 0) * a12 + vB(2, 0) * a22;
  A_hat(1, 1) = vB(0, 1) * a01 + vB(1, 1) * a11 + vB(2, 1) * a21;
  A_hat(1, 2) = vB(0, 1) * a02 + vB(1, 1) * a12 + vB(2, 1) * a22;
  A_hat(2, 2) = vB(0, 2) * a02 + vB(1, 2) * a12 + vB(2, 2) * a22;

  A_hat(1, 0) = A_hat(0, 1);
  A_hat(2, 0) = A_hat(0, 2);
  A_hat(2, 1) = A_hat(1, 2);

  /* Compute first m eigenpair of A_hat */
  View2D<T> vA("eigenvectors of A_hat", 3, m);
  syevx3x3_analytical(A_hat.data(), m, wp, vA.data());

  /* Compute eigenvectors ¦µ <- ¦µB_hat * ¦µA */
  View2D<T> v(vp, 3, m);

  v(0, 0) = vB(0, 0) * vA(0, 0) + vB(0, 1) * vA(1, 0) + vB(0, 2) * vA(2, 0);
  v(1, 0) = vB(1, 0) * vA(0, 0) + vB(1, 1) * vA(1, 0) + vB(1, 2) * vA(2, 0);
  v(2, 0) = vB(2, 0) * vA(0, 0) + vB(2, 1) * vA(1, 0) + vB(2, 2) * vA(2, 0);

  if (m > 1) {
    v(0, 1) = vB(0, 0) * vA(0, 1) + vB(0, 1) * vA(1, 1) + vB(0, 2) * vA(2, 1);
    v(1, 1) = vB(1, 0) * vA(0, 1) + vB(1, 1) * vA(1, 1) + vB(1, 2) * vA(2, 1);
    v(2, 1) = vB(2, 0) * vA(0, 1) + vB(2, 1) * vA(1, 1) + vB(2, 2) * vA(2, 1);
  }

  if (m > 2) {
    v(0, 2) = vB(0, 0) * vA(0, 2) + vB(0, 1) * vA(1, 2) + vB(0, 2) * vA(2, 2);
    v(1, 2) = vB(1, 0) * vA(0, 2) + vB(1, 1) * vA(1, 2) + vB(1, 2) * vA(2, 2);
    v(2, 2) = vB(2, 0) * vA(0, 2) + vB(2, 1) * vA(1, 2) + vB(2, 2) * vA(2, 2);
  }
}

/**
 * Given a real symmetric 2x2 matrix A, compute selected eigenvalues and eigenvectors use analytical
 * method
 */
template <typename T>
void syevx2x2_analytical(T* Ap, int m, T* wp, T* vp, bool verbose = true) {
  constexpr auto pi = Kokkos::numbers::pi_v<T>;
  View2D<T> A(Ap, 2, 2);  // RowMajor or ColMajor are the same since A is symmetric
  View1D<T> w(wp, m);
  View2D<T> v(vp, 2, m);  // RowMajor for host, ColMajor for device

  T a00 = A(0, 0);
  T a01 = A(0, 1);
  T a11 = A(1, 1);

  /* Check if matrix is diagonal */
  if (a01 * a01 == 0) {
    for (int i = 0; i < m; ++i) {
      w(i) = A(i, i);  // eigenvalues are diagonal elements
      for (int j = 0; j < m; ++j) {
        v(j, i) = (i == j) ? 1 : 0;  // eigenvectors are the identity matrix
      }
    }
    return;
  }

  /* Compute eigenvalues, the eigenvalues satisfy ¦Ë0 <= ¦Ë1 */
  T trA = a00 + a11;
  T detA = a00 * a11 - a01 * a01;
  T gapA = sqrt(trA * trA - 4 * detA);

  T w0 = (trA - gapA) / 2;
  T w1 = (trA + gapA) / 2;

  /* Compute eigenvectors */
  v(0, 0) = 1 / sqrt(1 + (w0 - a00) * (w0 - a00) / (a01 * a01));
  v(1, 0) = v(0, 0) * (w0 - a00) / a01;
  w(0) = w0;

  if (m > 1) {
    v(0, 1) = 1 / sqrt(1 + (w1 - a00) * (w1 - a00) / (a01 * a01));
    v(1, 1) = v(0, 1) * (w1 - a00) / a01;
    w(1) = w1;
  }
}

/**
 * Computes selected eigenpairs for 2x2 real generalized symmetric-definite eigenproblem Ax=¦ËBx
 */
template <typename T>
void sygvx2x2(T* Ap, T* Bp, int m, T* wp, T* vp, bool verbose = true) {
  View2D<T> A(Ap, 2, 2);
  View2D<T> B(Bp, 2, 2);

  /* Compute eigenvalues and eigenvectors of B */
  View2D<T> vB("eigenvectors of B", 2, 2);
  View1D<T> wB("eigenvalues of B", 2);
  syevx2x2_analytical(Bp, 2, wB.data(), vB.data());

  /* Compute ¦µB_hat = ¦µB * (¦«B^(1/2) + ¦ÅI)^(?1), in case ¦«B^(1/2) is singular */
  T eps = std::numeric_limits<T>::epsilon();
  wB(0) = 1 / (sqrt(wB(0)) + eps);
  wB(1) = 1 / (sqrt(wB(1)) + eps);

  vB(0, 0) *= wB(0);
  vB(1, 0) *= wB(0);

  vB(0, 1) *= wB(1);
  vB(1, 1) *= wB(1);

  /* Compute A_hat = ¦µB_hat * A * ¦µB_hat */
  View2D<T> A_hat("A_hat", 2, 2);
  T a00 = A(0, 0) * vB(0, 0) + A(0, 1) * vB(1, 0);
  T a10 = A(1, 0) * vB(0, 0) + A(1, 1) * vB(1, 0);

  T a01 = A(0, 0) * vB(0, 1) + A(0, 1) * vB(1, 1);
  T a11 = A(1, 0) * vB(0, 1) + A(1, 1) * vB(1, 1);

  A_hat(0, 0) = vB(0, 0) * a00 + vB(1, 0) * a10;
  A_hat(0, 1) = vB(0, 0) * a01 + vB(1, 0) * a11;
  A_hat(1, 1) = vB(0, 1) * a01 + vB(1, 1) * a11;
  A_hat(1, 0) = A_hat(0, 1);

  /* Compute first m eigenpair of A_hat */
  View2D<T> vA("eigenvectors of A_hat", 2, m);
  syevx2x2_analytical(A_hat.data(), m, wp, vA.data());

  /* Compute eigenvectors ¦µ <- ¦µB_hat * ¦µA */
  View2D<T> v(vp, 2, m);

  v(0, 0) = vB(0, 0) * vA(0, 0) + vB(0, 1) * vA(1, 0);
  v(1, 0) = vB(1, 0) * vA(0, 0) + vB(1, 1) * vA(1, 0);

  if (m > 1) {
    v(0, 1) = vB(0, 0) * vA(0, 1) + vB(0, 1) * vA(1, 1);
    v(1, 1) = vB(1, 0) * vA(0, 1) + vB(1, 1) * vA(1, 1);
  }
}

template <typename T>
void lobpcg(T* Ap, T* Bp, int n, int m, T* wp, T* vp, T* Xp = nullptr, T* Mp = nullptr,
            double tol = 1e-8, int maxiter = 500, bool verbose = true) {
  // TODO: Bp is nullptr, use identity matrix
  // TODO: sygv for full matrix eigenvalue problem
  // TODO: if not converged, pick up the result for lowest residual
  // TODO: if the lowest residual is still above 1e-4, then fail
  // TODO: print residual precision based on tolerance set
  // TODO: Xp need to copy to device

  const int m0 = m;                  // number of eigenpairs desired
  const int m1 = int(ceil(1 * m0));  // added number of eigenpairs to compute
  m = m0 + m1;                       // total number of eigenpairs to compute
  m = n < m ? n : m;                 // m cannot be larger than n

  /* If n < 100, use preconditioned sygvx */
  // if (n < 200) {
  //   m = n;
  // }

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

#ifdef KOKKOS_ENABLE_CUDA  // store in hstack [X, AX, BX, P, AP, BP, W, AW, BW] for device
  View2D<T> ABXPW("hstack: [X, AX, BX, P, AP, BP, W, AW, BW]", n, 9 * m);

  auto X_AX_BX = Kokkos::subview(ABXPW, Kokkos::ALL(), Kokkos::make_pair(0 * m, 3 * m));
  auto P_AP_BP = Kokkos::subview(ABXPW, Kokkos::ALL(), Kokkos::make_pair(3 * m, 6 * m));
  auto W_AW_BW = Kokkos::subview(ABXPW, Kokkos::ALL(), Kokkos::make_pair(6 * m, 9 * m));

  auto X = Kokkos::subview(X_AX_BX, Kokkos::ALL(), Kokkos::make_pair(0, m));
  auto W = Kokkos::subview(W_AW_BW, Kokkos::ALL(), Kokkos::make_pair(0, m));
  auto P = Kokkos::subview(P_AP_BP, Kokkos::ALL(), Kokkos::make_pair(0, m));

  auto AX = Kokkos::subview(X_AX_BX, Kokkos::ALL(), Kokkos::make_pair(m, 2 * m));
  auto AW = Kokkos::subview(W_AW_BW, Kokkos::ALL(), Kokkos::make_pair(m, 2 * m));
  auto AP = Kokkos::subview(P_AP_BP, Kokkos::ALL(), Kokkos::make_pair(m, 2 * m));

  auto BX = Kokkos::subview(X_AX_BX, Kokkos::ALL(), Kokkos::make_pair(2 * m, 3 * m));
  auto BW = Kokkos::subview(W_AW_BW, Kokkos::ALL(), Kokkos::make_pair(2 * m, 3 * m));
  auto BP = Kokkos::subview(P_AP_BP, Kokkos::ALL(), Kokkos::make_pair(2 * m, 3 * m));

  auto ABXP = Kokkos::subview(ABXPW, Kokkos::ALL(), Kokkos::make_pair(0, 6 * m));
  View2D<T> ABXP0("last iteration: ABXP", n, 6 * m);

#else  // store in vstack [X, AX, BX, P, AP, BP, W, AW, BW] for host
  View2D<T> ABXPW("vstack: [X, AX, BX, P, AP, BP, W, AW, BW]", 9 * n, m);

  auto X_AX_BX = Kokkos::subview(ABXPW, Kokkos::make_pair(0 * n, 3 * n), Kokkos::ALL());
  auto P_AP_BP = Kokkos::subview(ABXPW, Kokkos::make_pair(3 * n, 6 * n), Kokkos::ALL());
  auto W_AW_BW = Kokkos::subview(ABXPW, Kokkos::make_pair(6 * n, 9 * n), Kokkos::ALL());

  auto X = Kokkos::subview(X_AX_BX, Kokkos::make_pair(0, n), Kokkos::ALL());
  auto W = Kokkos::subview(W_AW_BW, Kokkos::make_pair(0, n), Kokkos::ALL());
  auto P = Kokkos::subview(P_AP_BP, Kokkos::make_pair(0, n), Kokkos::ALL());

  auto AX = Kokkos::subview(X_AX_BX, Kokkos::make_pair(n, 2 * n), Kokkos::ALL());
  auto AW = Kokkos::subview(W_AW_BW, Kokkos::make_pair(n, 2 * n), Kokkos::ALL());
  auto AP = Kokkos::subview(P_AP_BP, Kokkos::make_pair(n, 2 * n), Kokkos::ALL());

  auto BX = Kokkos::subview(X_AX_BX, Kokkos::make_pair(2 * n, 3 * n), Kokkos::ALL());
  auto BW = Kokkos::subview(W_AW_BW, Kokkos::make_pair(2 * n, 3 * n), Kokkos::ALL());
  auto BP = Kokkos::subview(P_AP_BP, Kokkos::make_pair(2 * n, 3 * n), Kokkos::ALL());

  auto ABXP = Kokkos::subview(ABXPW, Kokkos::make_pair(0, 6 * n), Kokkos::ALL());
  View2D<T> ABXP0("last iteration: ABXP", 6 * n, m);
#endif

  /* Compute XAX = X.T * A * X, XBX = X.T * B * X */
  Kokkos::DualView<T**> XAX("XAX", m, m);
  Kokkos::DualView<T**> XBX("XBX", m, m);

  if (Xp == nullptr) {
    Kokkos::parallel_for(
        "lobpcg::set: XAX, XBX", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {m, m}),
        KOKKOS_LAMBDA(int i, int j) {
          XAX.d_view(i, j) = A(i, j);  // X = eye(n, m) -> XAX = A[:m, :m]
          XBX.d_view(i, j) = B(i, j);  // X = eye(n, m) -> XBX = B[:m, :m]
        });
  } else {
    X = View2D<T>(Xp, n, m);
    KokkosBlas::gemm("N", "N", 1.0, A, X, 0.0, AX);           // AX = A * X
    KokkosBlas::gemm("N", "N", 1.0, B, X, 0.0, BX);           // BX = B * X
    KokkosBlas::gemm("T", "N", 1.0, X, AX, 0.0, XAX.d_view);  // XAX = X.T * AX
    KokkosBlas::gemm("T", "N", 1.0, X, BX, 0.0, XBX.d_view);  // XBX = X.T * BX
  }
  XAX.modify_device();  // mark XAX as modified on device
  XBX.modify_device();  // mark XBX as modified on device
  XAX.sync_host();      // sync XAX from device to host
  XBX.sync_host();      // sync XBX from device to host

  // printmat("XAX", XAX.h_view.data(), m, m);

  /* Solve generalized eigenvalue problem to get initial eigenvectors */
  Kokkos::DualView<T*> w("eigenvalues", m);
  Kokkos::DualView<T**> v("eigenvectors column major", m, m);

  lapackage::sygvx<T>(XAX.h_view.data(), XBX.h_view.data(), m, m, w.h_view.data(), v.h_view.data());

  w.modify_host();  // mark w as modified on host
  v.modify_host();  // mark v as modified on host
  w.sync_device();  // sync w from host to device
  v.sync_device();  // sync v from host to device

  // printmat("w", w.h_view.data(), 1, m);
  // printmat("v", v.h_view.data(), m, m, 1);

  // for (int i = 0; i < m; i++) {
  //   for (int j = 0; j < m; j++) {
  //     printf("v(%d, %d) = %f\n", i, j, v.h_view.data()[i + j * m]);
  //   }
  // }

  // Kokkos::parallel_for(
  //     "lobpcg::set: X = v", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {m, m}),
  //     KOKKOS_LAMBDA(int i, int j) {
  //       printf("v(%d, %d) = %f\n", i, j, v.d_view.data()[i + j * m]);
  //     });

  /* Compute: X = X * v, AX = A * X * v, BX = B * X * v */
  char TRANS = 'N';
#ifdef KOKKOS_ENABLE_CUDA
  TRANS = 'T';
#endif
  if (Xp == nullptr) {
    auto A_nm = Kokkos::subview(A, Kokkos::ALL(), Kokkos::make_pair(0, m));
    auto B_nm = Kokkos::subview(B, Kokkos::ALL(), Kokkos::make_pair(0, m));
    Kokkos::parallel_for(
        "X = eye(n, m) -> X = hstack(v, 0)",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {m, m}),
        KOKKOS_LAMBDA(int i, int j) { X(i, j) = v.d_view(j, i); });
    KokkosBlas::gemm("N", "TRANS", 1.0, A_nm, v.d_view, 0.0, AX);  // AX = A * v
    KokkosBlas::gemm("N", "TRANS", 1.0, B_nm, v.d_view, 0.0, BX);  // BX = B * v
  } else {
    View2D<T> X0("last X", n, m);
    Kokkos::deep_copy(X0, X);
    KokkosBlas::gemm("N", "N", 1.0, A, X, 0.0, AX);                        // AX = A * X
    KokkosBlas::gemm("N", "N", 1.0, B, X, 0.0, BX);                        // BX = B * X
    KokkosBlas::gemm("N", "TRANS", 1.0, X_AX_BX, v.d_view, 0.0, X_AX_BX);  // X = X * v
  }

  View2D<T> M;
  if (Mp != nullptr) {
    M = View2D<T>(Mp, n, n);
  }

  /* Compute residual: R = AX - BX * w, A_norm, B_norm */
  // HostView2D<T> h_R("h_R", n, m);

  View1D<T> neg_one("all->1", m);
  Kokkos::deep_copy(neg_one, -1.0);
  View2D<T> R("R = AX - BX * w", n, m);
  View2D<T> R_("R_ = AX + BX * w", n, m);
  // Kokkos::deep_copy(h_R, R);
  // printmat("R", h_R.data(), n, m);
  Kokkos::deep_copy(R, AX);  // R = AX
  // Kokkos::deep_copy(h_R, R);
  // printmat("R", h_R.data(), n, m);
  KokkosBlas::axpby(w.d_view, BX, neg_one, R);  // R = R - BX * w

  /* Initial norm array for [Xi, Wi, Pi] */
  View2D<T> norm("norm", m, 3);
  Kokkos::deep_copy(norm, 1.0);
  for (int i = 0; i < m; i++) {
    norm(i, 0) = sqrt(KokkosBlas::nrm2_squared(Kokkos::subview(X, Kokkos::ALL(), i)));
    norm(i, 1) = sqrt(KokkosBlas::nrm2_squared(Kokkos::subview(R, Kokkos::ALL(), i)));
  }

  // Kokkos::deep_copy(h_R, R);
  // printmat("R", h_R.data(), n, m, 1);
  // printf("h_R(2, 0) = %f\n", h_R(2, 0));

/* Initial for outer and inner loop */
#ifdef KOKKOS_ENABLE_CUDA
  Kokkos::DualView<T**> gramAB_outer("hstack: [gramA_outer, gramB_outer]", m, 2 * m);
  auto gramA_outer = Kokkos::subview(gramAB_outer, Kokkos::ALL(), Kokkos::make_pair(0, m));
  auto gramB_outer = Kokkos::subview(gramAB_outer, Kokkos::ALL(), Kokkos::make_pair(m, 2 * m));

  View2D<T> gramAB_inner("hstack: [gramA_inner, gramB_inner]", 3, 6);
  auto gramA_inner = Kokkos::subview(gramAB_inner, Kokkos::ALL(), Kokkos::make_pair(0, 3));
  auto gramB_inner = Kokkos::subview(gramAB_inner, Kokkos::ALL(), Kokkos::make_pair(3, 6));

  View2D<T> S("hstack: [Xi, Wi, Pi], m sub-blocks[n, 3]", n, 3 * m);
  View2D<T> ABS("hstack: [AXi, AWi, APi, BXi, BWi, BPi], m sub-blocks[n, 6]", n, 6 * m);
#else
  Kokkos::DualView<T**> gramAB_outer("vstack: [gramA_outer, gramB_outer]", 2 * m, m);
  auto gramA_outer = Kokkos::subview(gramAB_outer, Kokkos::make_pair(0, m), Kokkos::ALL());
  auto gramB_outer = Kokkos::subview(gramAB_outer, Kokkos::make_pair(m, 2 * m), Kokkos::ALL());

  View2D<T> gramAB_inner("vstack: [gramA_inner, gramB_inner]", 6, 3);
  auto gramA_inner = Kokkos::subview(gramAB_inner, Kokkos::make_pair(0, 3), Kokkos::ALL());
  auto gramB_inner = Kokkos::subview(gramAB_inner, Kokkos::make_pair(3, 6), Kokkos::ALL());

  View2D<T> S("vstack: [Xi, Wi, Pi], m sub-blocks[n, 3]", m * n, 3);
  View2D<T> ABS("vstack: [AXi, AWi, APi, BXi, BWi, BPi], m sub-blocks[n, 6]", m * n, 6);
#endif

  Kokkos::DualView<T*> w_outer("outer eigenvalues", m);
  Kokkos::DualView<T**> v_outer("outer eigenvectors", m, m);
  View1D<T> w_inner("inner eigenvalues", 1);
  View1D<T> v_inner("inner eigenvectors", 3);

  View1D<T> v0("v0", m);
  View1D<T> v1("v1", m);
  View1D<T> v2("v2", m);

  /* Initial convergent array as all false: 0 in host */
  Kokkos::DualView<T*> is_convergent("convergent flag", m);

  /* Start outer loop */
  for (int k = 0; k < maxiter; k++) {
    if (Mp != nullptr) {                              // with preconditioning, normally M = A^-1
      KokkosBlas::gemm("N", "N", 1.0, M, R, 0.0, W);  // W = M * R
    } else {                                          // without preconditioning
      Kokkos::deep_copy(W, R);
    }

    // printmat("W", W.data(), n, 3);
    // printf("norm(0, 1) = %f\n", norm(0, 1));
    // printf("norm(1, 1) = %f\n", norm(1, 1));

    if (k == 1) {
      KokkosBlas::gemm("N", "N", 1.0, A, P, 0.0, AP);
      KokkosBlas::gemm("N", "N", 1.0, B, P, 0.0, BP);
    }

    KokkosBlas::gemm("N", "N", 1.0, A, W, 0.0, AW);
    KokkosBlas::gemm("N", "N", 1.0, B, W, 0.0, BW);

    /* Perform Rayleigh-Ritz procedure */
    /* Normalize [Xi, Wi, Pi] and store them in contiguous memory for each sub-block */
#ifdef KOKKOS_ENABLE_CUDA
    Kokkos::parallel_for(
        "hstack: S = [Xi, Wi, Pi], m sub-blocks[n, 3]"
        "hstack: ABS = [AXi, AWi, APi, BXi, BWi, BPi], m sub-blocks[n, 6]",
        Kokkos::RangePolicy<ExecSpace>(0, n), KOKKOS_LAMBDA(const int i) {
          for (int j = 0; j < m; ++j) {  // memory contiguity in row, i.e. along m
            if (is_convergent.d_view(j) == 0) {
              T X_norm = norm(j, 0);
              T W_norm = norm(j, 1);
              T P_norm = norm(j, 2);

              S(i, 0 + 3 * j) = X(i, j) = X(i, j) / X_norm;
              S(i, 1 + 3 * j) = W(i, j) = W(i, j) / W_norm;
              S(i, 2 + 3 * j) = P(i, j) = P(i, j) / P_norm;

              ABS(i, 0 + 6 * j) = AX(i, j) = AX(i, j) / X_norm;
              ABS(i, 1 + 6 * j) = AW(i, j) = AW(i, j) / W_norm;
              ABS(i, 2 + 6 * j) = AP(i, j) = AP(i, j) / P_norm;
              ABS(i, 3 + 6 * j) = BX(i, j) = BX(i, j) / X_norm;
              ABS(i, 4 + 6 * j) = BW(i, j) = BW(i, j) / W_norm;
              ABS(i, 5 + 6 * j) = BP(i, j) = BP(i, j) / P_norm;
            }
          }
        });
#else
    Kokkos::parallel_for(
        "vstack: S = [Xi, Wi, Pi], m sub-blocks[n, 3]"
        "vstack: ABS = [AXi, AWi, APi, BXi, BWi, BPi], m sub-blocks[n, 6]",
        Kokkos::RangePolicy<ExecSpace>(0, n), KOKKOS_LAMBDA(const int i) {
          for (int j = 0; j < m; ++j) {  // memory contiguity in row, i.e. along m
            if (is_convergent.d_view(j) == 0) {
              T X_norm = norm(j, 0);
              T W_norm = norm(j, 1);
              T P_norm = norm(j, 2);

              S(i + n * j, 0) = X(i, j) = X(i, j) / X_norm;
              S(i + n * j, 1) = W(i, j) = W(i, j) / W_norm;
              S(i + n * j, 2) = P(i, j) = P(i, j) / P_norm;

              ABS(i + n * j, 0) = AX(i, j) = AX(i, j) / X_norm;
              ABS(i + n * j, 1) = AW(i, j) = AW(i, j) / W_norm;
              ABS(i + n * j, 2) = AP(i, j) = AP(i, j) / P_norm;
              ABS(i + n * j, 3) = BX(i, j) = BX(i, j) / X_norm;
              ABS(i + n * j, 4) = BW(i, j) = BW(i, j) / W_norm;
              ABS(i + n * j, 5) = BP(i, j) = BP(i, j) / P_norm;
            }
          }
        });
#endif
    // tick("inner loop");

    /* Perform inner Rayleigh-Ritz procedure */
    /* Use hard lock technique to lock the convergent eigenpairs */
    for (int i = 0; i < m; i++) {
      if (is_convergent.h_view(i) == 0) {  // only loop over non convergent eigenpairs
#ifdef KOKKOS_ENABLE_CUDA
        /* Compute symmetric Gram matrices */
        auto Si = Kokkos::subview(S, Kokkos::ALL(), Kokkos::make_pair(i * 3, i * 3 + 3));
        auto ABSi = Kokkos::subview(ABS, Kokkos::ALL(), Kokkos::make_pair(i * 6, i * 6 + 6));
        KokkosBlas::gemm("T", "N", 1.0, Si, ABSi, 0.0, gramAB_inner);

        // HostView2D<T> gramAB_inner_h("gramAB_inner_h", 3, 6);
        // Kokkos::deep_copy(gramAB_inner_h, gramAB_inner.d_view);
        // printmat("gramAB_inner_h", gramAB_inner_h.data(), 3, 6, 1);

        // HostView2D<T> h_Si("h_Si", n, 3);
        // HostView2D<T> h_ABSi("h_ABSi", n, 6);
        // Kokkos::deep_copy(h_Si, Si);
        // Kokkos::deep_copy(h_ABSi, ABSi);
        // printmat("h_Si", h_Si.data(), n, 3, 1);
        // printmat("h_ABSi", h_ABSi.data(), n, 6, 1);
#else
        /* Compute symmetric Gram matrices */
        auto Si = Kokkos::subview(S, Kokkos::make_pair(i * n, (i + 1) * n), Kokkos::ALL());
        auto ABSi = Kokkos::subview(ABS, Kokkos::make_pair(i * n, (i + 1) * n), Kokkos::ALL());
        KokkosBlas::gemm("T", "N", 1.0, ABSi, Si, 0.0, gramAB_inner);
#endif

        // gramAB_inner.modify_device();
        // gramAB_inner.sync_host();

        // printmat("gramA_inner", gramA_inner.data(), 3, 3);
        // printmat("gramB_inner", gramB_inner.data(), 3, 3);

        /* Make sure store gramA, gramB to contigous memory */
        int n_inner = (k == 0) ? 2 : 3;
#ifdef KOKKOS_ENABLE_CUDA
        if (k == 0) {
          gramA_inner(2, 0) = gramA_inner(0, 1);
          gramA_inner(0, 1) = gramA_inner(1, 1);
          gramB_inner(2, 0) = gramB_inner(0, 1);
          gramB_inner(0, 1) = gramB_inner(1, 1);
        }
#else
        if (k == 0) {
          gramA_inner(0, 2) = gramA_inner(1, 0);
          gramA_inner(1, 0) = gramA_inner(1, 1);
          gramB_inner(0, 2) = gramB_inner(1, 0);
          gramB_inner(1, 0) = gramB_inner(1, 1);
        }
#endif
        // printmat("gramA_inner", gramA_inner.data(), 3, 3);
        // printmat("gramB_inner", gramB_inner.data(), 3, 3);

        /* Compute eigenvalues and eigenvectors 3x3 eigenvalue problem */
        // lapackage::sygvx<T>(gramA_inner.data(), gramB_inner.data(), n_inner, 1, w_inner.data(),
        //                     v_inner.data());

        if (k == 0) {
          sygvx2x2(gramA_inner.data(), gramB_inner.data(), 1, w_inner.data(), v_inner.data());
        } else {
          sygvx3x3(gramA_inner.data(), gramB_inner.data(), 1, w_inner.data(), v_inner.data());
        }

        /* Only store the result, move the computation out of the loop */
        v0(i) = v_inner(0);
        v1(i) = v_inner(1);
        v2(i) = v_inner(2);
      }
    }
    // tock("inner loop");

    // v0.modify_host();
    // v1.modify_host();
    // v2.modify_host();
    // v0.sync_device();
    // v1.sync_device();
    // v2.sync_device();

// printmat("v0", v0.h_view.data(), m, 1);
// printmat("v1", v1.h_view.data(), m, 1);
// printmat("v2", v2.h_view.data(), m, 1);

/* Compute the Ritz vector, compute batchly out of inner loop */
#ifdef KOKKOS_ENABLE_CUDA
    View1D<T> one("all->1", 3 * m);
    Kokkos::deep_copy(one, 1.0);
    View1D<T> v0_v0_v0("v0_v0_v0", 3 * m);
    View1D<T> v1_v1_v1("v1_v1_v1", 3 * m);
    View1D<T> v2_v2_v2("v2_v2_v2", 3 * m);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecSpace>(0, m), KOKKOS_LAMBDA(const int i) {
          v0_v0_v0(i) = v0(i);
          v1_v1_v1(i) = v1(i);
          v2_v2_v2(i) = v2(i);

          v0_v0_v0(i + m) = v0(i);
          v1_v1_v1(i + m) = v1(i);
          v2_v2_v2(i + m) = v2(i);

          v0_v0_v0(i + 2 * m) = v0(i);
          v1_v1_v1(i + 2 * m) = v1(i);
          v2_v2_v2(i + 2 * m) = v2(i);
        });

    KokkosBlas::axpby(v1_v1_v1, W_AW_BW, v2_v2_v2, P_AP_BP);  // P = W * v(1) + P * v(2)
    KokkosBlas::axpby(one, P_AP_BP, v0_v0_v0, X_AX_BX);       // X = X * v(0) + P
#else
    View1D<T> one("all->1", m);
    Kokkos::deep_copy(one, 1.0);
    KokkosBlas::axpby(v1, W_AW_BW, v2, P_AP_BP);   // P = W * v(1) + P * v(2)
    KokkosBlas::axpby(one, P_AP_BP, v0, X_AX_BX);  // X = X * v(0) + P
#endif

    // HostView2D<T> h_P_AP_BP("h_P_AP_BP", n, 3*m);
    // HostView2D<T> h_X_AX_BX("h_X_AX_BX", n, 3*m);
    // Kokkos::deep_copy(h_P_AP_BP, P_AP_BP);
    // Kokkos::deep_copy(h_X_AX_BX, X_AX_BX);
    // printmat("h_P_AP_BP", h_P_AP_BP.data(), n, 3*m,1);
    // printmat("h_X_AX_BX", h_X_AX_BX.data(), n, 3*m,1);

    /* Perform outer Rayleigh-Ritz procedure */
    KokkosBlas::gemm("T", "N", 1.0, X, AX, 0.0, gramA_outer.d_view);  // gramA = X^T * AX
    KokkosBlas::gemm("T", "N", 1.0, X, BX, 0.0, gramB_outer.d_view);  // gramB = X^T * BX

    /* Compute eigenvalues and eigenvectors for m x m eigenvalue problem */
    gramAB_outer.modify_device();
    gramAB_outer.sync_host();

    // printmat("gramA_outer", gramA_outer.h_view.data(), m, m,1);
    // printmat("gramB_outer", gramB_outer.h_view.data(), m, m,1);

    lapackage::sygvx<T>(gramA_outer.h_view.data(), gramB_outer.h_view.data(), m, m,
                        w_outer.h_view.data(), v_outer.h_view.data());

    w_outer.modify_host();
    v_outer.modify_host();
    w_outer.sync_device();
    v_outer.sync_device();

    // printmat("w_outer", w_outer.h_view.data(), m, 1);
    // printmat("v_outer", v_outer.h_view.data(), m, m);

    /* [X, AX, BX, P, AP, BP] = [X, AX, BX, P, AP, BP] * v */
    Kokkos::deep_copy(ABXP0, ABXP);
#ifdef KOKKOS_ENABLE_CUDA
    auto X0 = Kokkos::subview(ABXP0, Kokkos::ALL(), Kokkos::make_pair(0, m));
    auto AX0 = Kokkos::subview(ABXP0, Kokkos::ALL(), Kokkos::make_pair(m, 2 * m));
    auto BX0 = Kokkos::subview(ABXP0, Kokkos::ALL(), Kokkos::make_pair(2 * m, 3 * m));

    auto P0 = Kokkos::subview(ABXP0, Kokkos::ALL(), Kokkos::make_pair(3 * m, 4 * m));
    auto AP0 = Kokkos::subview(ABXP0, Kokkos::ALL(), Kokkos::make_pair(4 * m, 5 * m));
    auto BP0 = Kokkos::subview(ABXP0, Kokkos::ALL(), Kokkos::make_pair(5 * m, 6 * m));

    KokkosBlas::gemm("N", "N", 1.0, X0, v_outer.d_view, 0.0, X);
    KokkosBlas::gemm("N", "N", 1.0, AX0, v_outer.d_view, 0.0, AX);
    KokkosBlas::gemm("N", "N", 1.0, BX0, v_outer.d_view, 0.0, BX);

    KokkosBlas::gemm("N", "N", 1.0, P0, v_outer.d_view, 0.0, P);
    KokkosBlas::gemm("N", "N", 1.0, AP0, v_outer.d_view, 0.0, AP);
    KokkosBlas::gemm("N", "N", 1.0, BP0, v_outer.d_view, 0.0, BP);
#else
    KokkosBlas::gemm("N", "T", 1.0, ABXP0, v_outer.d_view, 0.0, ABXP);
#endif

    /* Compute Residual: res = ||AX - BX * w|| / ||AX + BX * w|| */
    Kokkos::View<T*, Kokkos::HostSpace> res("residual norm stored in host", m);
    Kokkos::deep_copy(R, AX);                           // R = AX
    KokkosBlas::axpby(w_outer.d_view, BX, neg_one, R);  // R = BX * w - R
    KokkosBlas::update(1.0, R, 2.0, AX, 0.0, R_);       // R_ = R + 2*AX

    /* update residual norm, and norm of Xi, Wi, Pi */
    for (int i = 0; i < m; i++) {
      if (is_convergent.h_view(i) == 0) {
        if (i < m0) {
          auto Ri = Kokkos::subview(R, Kokkos::ALL(), i);
          auto Ri_ = Kokkos::subview(R_, Kokkos::ALL(), i);

          T Ri_norm = KokkosBlas::nrm2_squared(Ri);    // ||Ri||^2
          T Ri_norm_ = KokkosBlas::nrm2_squared(Ri_);  // ||Ri_||^2

          res(i) = sqrt(Ri_norm / Ri_norm_);  // res(i) = ||Ri|| / ||Ri_||
        }
        norm(i, 0) = sqrt(KokkosBlas::nrm2_squared(Kokkos::subview(X, Kokkos::ALL(), i)));
        norm(i, 1) = sqrt(KokkosBlas::nrm2_squared(Kokkos::subview(R, Kokkos::ALL(), i)));
        norm(i, 2) = sqrt(KokkosBlas::nrm2_squared(Kokkos::subview(P, Kokkos::ALL(), i)));
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
        is_convergent.h_view(i) = 0;  // not converged
      } else {
        is_convergent.h_view(i) = 1;  // converged
        count++;
      }
    }

    is_convergent.modify_host();  // mark is_convergent as modified on host
    is_convergent.sync_device();  // sync is_convergent from host to device

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
  // TODO: copy to host
  View1D<T> w_result(wp, m0);
  View2D<T> v_result(vp, n, m0);
  auto X_m0 = Kokkos::subview(X, Kokkos::ALL(), Kokkos::make_pair(0, m0));
  auto w_m0 = Kokkos::subview(w_outer.d_view, Kokkos::make_pair(0, m0));
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