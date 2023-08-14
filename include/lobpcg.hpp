#pragma once
#ifndef LOBPCG_HPP
#define LOBPCG_HPP

#include <KokkosBlas.hpp>
#include <KokkosBlas1_iamax.hpp>
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
 *    3. 'A' have to be normalised, otherwise the result will lose precision for case ��i >> ��j
 *
 * Reference:
 *    1. https://en.wikipedia.org/wiki/Eigenvalue_algorithm#cite_note-Smith-19
 *    2. Smith, Oliver K. (April 1961), "Eigenvalues of a symmetric 3 �� 3 matrix.", Communications
 * of the ACM, 4 (4): 168, doi:10.1145/355578.366316, S2CID 37815415
 */
template <typename T>
void syevx3x3_analytical(T* Ap, int m, T* wp, T* vp, bool verbose = true) {
  constexpr auto pi = Kokkos::numbers::pi_v<T>;
  HostView2D<T> A(Ap, 3, 3);  // RowMajor or ColMajor are the same since A is symmetric
  HostView1D<T> w(wp, m);
  HostView2D<T> v(vp, 3, m);  // RowMajor for host, ColMajor for device

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

  /* Compute eigenvalues, the eigenvalues satisfy ��0 <= ��1 <= ��2 */
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
 * Computes selected eigenpairs for 3x3 real generalized symmetric-definite eigenproblem Ax=��Bx
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
 *    3. 'A' have to be normalised, otherwise the result will lose precision for case ��i >> ��j
 *
 * Algorithm:
 *    1. ��B, ��B <- B * ��B = ��B * ��B
 *    2. ��B_hat <- ��B_hat = ��B * ��B^(?1/2) �� ��B * (��B^(1/2) + ��I)^(?1)
 *    3. A_hat <- A_hat = ��B_hat * A * ��B_hat
 *    4. ��A, ��A <- A_hat * ��A = ��A * ��A
 *    5. �� <- ��A, �� <- ��B_hat * ��A
 *
 * Reference:
 *    1. Ghojogh B, Karray F, Crowley M. Eigenvalue and generalized eigenvalue problems:
 * Tutorial[J]. arXiv preprint arXiv:1903.11240, 2019.
 */
template <typename T>
void sygvx3x3(T* Ap, T* Bp, int m, T* wp, T* vp, bool verbose = true) {
  HostView2D<T> A(Ap, 3, 3);
  HostView2D<T> B(Bp, 3, 3);

  /* Compute eigenvalues and eigenvectors of B */
  HostView2D<T> vB("eigenvectors of B", 3, 3);
  HostView1D<T> wB("eigenvalues of B", 3);
  syevx3x3_analytical(B.data(), 3, wB.data(), vB.data());

  /* Compute ��B_hat = ��B * (��B^(1/2) + ��I)^(?1), in case ��B^(1/2) is singular */
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

  /* Compute A_hat = ��B_hat * A * ��B_hat */
  HostView2D<T> A_hat("A_hat", 3, 3);
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
  HostView2D<T> vA("eigenvectors of A_hat", 3, m);
  syevx3x3_analytical(A_hat.data(), m, wp, vA.data());

  /* Compute eigenvectors �� <- ��B_hat * ��A */
  HostView2D<T> v(vp, 3, m);

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
  HostView2D<T> A(Ap, 2, 2);  // RowMajor or ColMajor are the same since A is symmetric
  HostView1D<T> w(wp, m);
  HostView2D<T> v(vp, 2, m);  // RowMajor for host, ColMajor for device

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

  /* Compute eigenvalues, the eigenvalues satisfy ��0 <= ��1 */
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
 * Computes selected eigenpairs for 2x2 real generalized symmetric-definite eigenproblem Ax=��Bx
 */
template <typename T>
void sygvx2x2(T* Ap, T* Bp, int m, T* wp, T* vp, bool verbose = true) {
  HostView2D<T> A(Ap, 2, 2);
  HostView2D<T> B(Bp, 2, 2);

  /* Compute eigenvalues and eigenvectors of B */
  HostView2D<T> vB("eigenvectors of B", 2, 2);
  HostView1D<T> wB("eigenvalues of B", 2);
  syevx2x2_analytical(B.data(), 2, wB.data(), vB.data());

  /* Compute ��B_hat = ��B * (��B^(1/2) + ��I)^(?1), in case ��B^(1/2) is singular */
  T eps = std::numeric_limits<T>::epsilon();
  wB(0) = 1 / (sqrt(wB(0)) + eps);
  wB(1) = 1 / (sqrt(wB(1)) + eps);

  vB(0, 0) *= wB(0);
  vB(1, 0) *= wB(0);

  vB(0, 1) *= wB(1);
  vB(1, 1) *= wB(1);

  /* Compute A_hat = ��B_hat * A * ��B_hat */
  HostView2D<T> A_hat("A_hat", 2, 2);
  T a00 = A(0, 0) * vB(0, 0) + A(0, 1) * vB(1, 0);
  T a10 = A(1, 0) * vB(0, 0) + A(1, 1) * vB(1, 0);

  T a01 = A(0, 0) * vB(0, 1) + A(0, 1) * vB(1, 1);
  T a11 = A(1, 0) * vB(0, 1) + A(1, 1) * vB(1, 1);

  A_hat(0, 0) = vB(0, 0) * a00 + vB(1, 0) * a10;
  A_hat(0, 1) = vB(0, 0) * a01 + vB(1, 0) * a11;
  A_hat(1, 1) = vB(0, 1) * a01 + vB(1, 1) * a11;
  A_hat(1, 0) = A_hat(0, 1);

  /* Compute first m eigenpair of A_hat */
  HostView2D<T> vA("eigenvectors of A_hat", 2, m);
  syevx2x2_analytical(A_hat.data(), m, wp, vA.data());

  /* Compute eigenvectors �� <- ��B_hat * ��A */
  HostView2D<T> v(vp, 2, m);

  v(0, 0) = vB(0, 0) * vA(0, 0) + vB(0, 1) * vA(1, 0);
  v(1, 0) = vB(1, 0) * vA(0, 0) + vB(1, 1) * vA(1, 0);

  if (m > 1) {
    v(0, 1) = vB(0, 0) * vA(0, 1) + vB(0, 1) * vA(1, 1);
    v(1, 1) = vB(1, 0) * vA(0, 1) + vB(1, 1) * vA(1, 1);
  }
}

template <typename T>
bool check_convergence(T* residual, T* is_convergent, const int m, const int k, const int maxiter,
                       const double tol, bool verbose = true) {
  T max_residual = 0.0;
  int count = 0;
  bool converged = false;

  for (int i = 0; i < m; i++) {
    max_residual = std::max(max_residual, residual[i]);

    if (residual[i] < tol) {
      is_convergent[i] = 1.0;
      count++;
    } else {
      is_convergent[i] = 0.0;
    }
  }

  if (verbose) {
    printf(
        "Iteration: \033[32m%2d\033[0m, converged: \033[32m%2d\033[0m, "
        "residual: \033[32m%e\033[0m\n",
        k, count, max_residual);
  }

  if (max_residual < tol || count == m) {
    converged = true;
  }

  if (k == maxiter - 1) {
    printf(
        "\033[1;31mWarning\033[0m: maximum number of iterations reached, "
        "residual = %e\n",
        max_residual);
  }

  return converged;
}

template <typename T>
void lobpcgI(T* Ap, T* Bp, int n, int m, T* wp, T* vp, T* Xp = nullptr, T* Mp = nullptr,
             double tol = 1e-8, int maxiter = 500, bool verbose = true) {
  // TODO: Bp is nullptr, use identity matrix
  // TODO: sygv for full matrix eigenvalue problem
  // TODO: if not converged, pick up the result for lowest residual
  // TODO: if the lowest residual is still above 1e-4, then fail
  // TODO: print residual precision based on tolerance set
  // TODO: Xp need to copy to device
  Xp = nullptr;
  const int m0 = m;                    // number of eigenpairs desired
  const int m1 = int(ceil(1.0 * m0));  // added number of eigenpairs to compute
  m = m0 + m1;                         // total number of eigenpairs to compute
  m = n < m ? n : m;                   // m cannot be larger than n

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

  /* Store in: [X W P | AX AW AP | BX BW BP] */
  View2D<T> S_AS_BS("vhstack: [X W P | AX AW AP | BX BW BP]", 3 * n, 3 * m);
  auto S = Kokkos::subview(S_AS_BS, Kokkos::make_pair(0, n), Kokkos::ALL());
  auto AS = Kokkos::subview(S_AS_BS, Kokkos::make_pair(n, 2 * n), Kokkos::ALL());
  auto BS = Kokkos::subview(S_AS_BS, Kokkos::make_pair(2 * n, 3 * n), Kokkos::ALL());

  auto X_AX_BX = Kokkos::subview(S_AS_BS, Kokkos::ALL(), Kokkos::make_pair(0, m));
  auto W_AW_BW = Kokkos::subview(S_AS_BS, Kokkos::ALL(), Kokkos::make_pair(m, 2 * m));
  auto P_AP_BP = Kokkos::subview(S_AS_BS, Kokkos::ALL(), Kokkos::make_pair(2 * m, 3 * m));

  auto X = Kokkos::subview(X_AX_BX, Kokkos::make_pair(0, n), Kokkos::ALL());
  auto W = Kokkos::subview(W_AW_BW, Kokkos::make_pair(0, n), Kokkos::ALL());
  auto P = Kokkos::subview(P_AP_BP, Kokkos::make_pair(0, n), Kokkos::ALL());

  auto AX = Kokkos::subview(X_AX_BX, Kokkos::make_pair(n, 2 * n), Kokkos::ALL());
  auto AW = Kokkos::subview(W_AW_BW, Kokkos::make_pair(n, 2 * n), Kokkos::ALL());
  auto AP = Kokkos::subview(P_AP_BP, Kokkos::make_pair(n, 2 * n), Kokkos::ALL());

  auto BX = Kokkos::subview(X_AX_BX, Kokkos::make_pair(2 * n, 3 * n), Kokkos::ALL());
  auto BW = Kokkos::subview(W_AW_BW, Kokkos::make_pair(2 * n, 3 * n), Kokkos::ALL());
  auto BP = Kokkos::subview(P_AP_BP, Kokkos::make_pair(2 * n, 3 * n), Kokkos::ALL());

  /* Compute XAX0 = X.T * A * X, XBX0 = X.T * B * X */
  View2D<T> XAX0("XAX0", m, m);
  View2D<T> XBX0("XBX0", m, m);
  if (Xp == nullptr) {
    Kokkos::parallel_for(
        "lobpcg::set: XAX0, XBX0",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {m, m}),
        KOKKOS_LAMBDA(int i, int j) {
          XAX0(i, j) = A(i, j);  // X = eye(n, m) -> XAX0 = A[:m, :m]
          XBX0(i, j) = B(i, j);  // X = eye(n, m) -> XBX0 = B[:m, :m]
        });
  } else {
    X = View2D<T>(Xp, n, m);
    KokkosBlas::gemm("N", "N", 1.0, A, X, 0.0, AX);     // AX = A * X
    KokkosBlas::gemm("N", "N", 1.0, B, X, 0.0, BX);     // BX = B * X
    KokkosBlas::gemm("T", "N", 1.0, X, AX, 0.0, XAX0);  // XAX0 = X.T * AX
    KokkosBlas::gemm("T", "N", 1.0, X, BX, 0.0, XBX0);  // XBX0 = X.T * BX
  }

  /* Solve generalized eigenvalue problem to get initial eigenvectors */
  View1D<T> w("eigenvalues", m);
  View2D<T> v("eigenvectors: column major", m, m);

  lapackage::sygvx<T>(XAX0.data(), XBX0.data(), m, m, w.data(), v.data());

  /* Compute: X = X * v, AX = A * X * v, BX = B * X * v */
  if (Xp == nullptr) {
    auto A_nm = Kokkos::subview(A, Kokkos::ALL(), Kokkos::make_pair(0, m));
    auto B_nm = Kokkos::subview(B, Kokkos::ALL(), Kokkos::make_pair(0, m));

    Kokkos::parallel_for(
        "X = eye(n, m) -> X = hstack(v, 0)",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {m, m}),
        KOKKOS_LAMBDA(int i, int j) { X(i, j) = v(j, i); });  // X = v.T

    KokkosBlas::gemm("N", "T", 1.0, A_nm, v, 0.0, AX);  // AX = A * X
    KokkosBlas::gemm("N", "T", 1.0, B_nm, v, 0.0, BX);  // BX = B * X
  } else {
    View2D<T> X0 = X;
    KokkosBlas::gemm("N", "N", 1.0, A, X, 0.0, AX);             // AX = A * X
    KokkosBlas::gemm("N", "N", 1.0, B, X, 0.0, BX);             // BX = B * X
    KokkosBlas::gemm("N", "T", 1.0, X_AX_BX, v, 0.0, X_AX_BX);  // X = X * v
  }

  View2D<T> M;
  if (Mp != nullptr) {
    M = View2D<T>(Mp, n, n);
  }

  /* Compute residual: R = AX - BX * w, A_norm, B_norm */
  View2D<T> R("R = AX - BX * w", n, m);
  Kokkos::parallel_for(
      "R = AX - BX * w", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, m}),
      KOKKOS_LAMBDA(int i, int j) { R(i, j) = AX(i, j) - BX(i, j) * w(j); });

  /* Initial: norm array for [Xi, Wi, Pi] */
  View2D<T> norm("norm", m, 3);
  View2D<T> norm2("norm", m0, 2);
  KokkosBlas::fill(norm, 1.0);  // initialize norm to 0

  /* Initial: gram matrix for Rayleigh-Ritz */
  View2D<T> gramA("gramA", 3 * m, 3 * m);
  View2D<T> gramB("gramB", 3 * m, 3 * m);

  /* Initial: convergent array as all false: 0 in host */
  View1D<T> is_convergent("convergent flag", m);
  View1D<T> res("residual norm stored in host", m);

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

    /* Normalize: X, W, P, AX, AW, AP, BX, BW, BP */
    Kokkos::parallel_for(
        "Xi_norm", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, m}),
        KOKKOS_LAMBDA(const int i, const int j) {
          if (is_convergent(j) == 0) {
            Kokkos::atomic_add(&norm(j, 0), X(i, j) * X(i, j));
            Kokkos::atomic_add(&norm(j, 1), W(i, j) * W(i, j));
            Kokkos::atomic_add(&norm(j, 2), P(i, j) * P(i, j));
          }
        });

    Kokkos::parallel_for(
        3 * n, KOKKOS_LAMBDA(const int i) {
          for (int j = 0; j < m; ++j) {  // memory contiguity in row, i.e. along m
            if (is_convergent(j) == 0) {
              X_AX_BX(i, j) /= sqrt(norm(j, 0));
              W_AW_BW(i, j) /= sqrt(norm(j, 1));
              if (k != 0) {
                P_AP_BP(i, j) /= sqrt(norm(j, 2));
              }
            }
          }
        });

    /* Perform Rayleigh-Ritz procedure */
    if (m < 150) {
      KokkosBlas::gemm("T", "N", 1.0, S, AS, 0.0, gramA);  // gramA = S^T * AS
      KokkosBlas::gemm("T", "N", 1.0, S, BS, 0.0, gramB);  // gramB = S^T * BS
    } else {
      auto pair_0_m = Kokkos::make_pair(0, m);
      auto pair_m_2m = Kokkos::make_pair(m, 2 * m);
      auto pair_2m_3m = Kokkos::make_pair(2 * m, 3 * m);
      auto XAX = Kokkos::subview(gramA, pair_0_m, pair_0_m);
      auto XBX = Kokkos::subview(gramB, pair_0_m, pair_0_m);
      auto XAW = Kokkos::subview(gramA, pair_0_m, pair_m_2m);
      auto XBW = Kokkos::subview(gramB, pair_0_m, pair_m_2m);
      auto XAP = Kokkos::subview(gramA, pair_0_m, pair_2m_3m);
      auto XBP = Kokkos::subview(gramB, pair_0_m, pair_2m_3m);
      auto WAW = Kokkos::subview(gramA, pair_m_2m, pair_m_2m);
      auto WBW = Kokkos::subview(gramB, pair_m_2m, pair_m_2m);
      auto WAP = Kokkos::subview(gramA, pair_m_2m, pair_2m_3m);
      auto WBP = Kokkos::subview(gramB, pair_m_2m, pair_2m_3m);
      auto PAP = Kokkos::subview(gramA, pair_2m_3m, pair_2m_3m);
      auto PBP = Kokkos::subview(gramB, pair_2m_3m, pair_2m_3m);

      KokkosBlas::gemm("T", "N", 1.0, X, AX, 0.0, XAX);  // XAX = X^T * AX
      KokkosBlas::gemm("T", "N", 1.0, X, BX, 0.0, XBX);  // XBX = X^T * BX
      KokkosBlas::gemm("T", "N", 1.0, X, AW, 0.0, XAW);  // XAW = X^T * AW
      KokkosBlas::gemm("T", "N", 1.0, X, BW, 0.0, XBW);  // XBW = X^T * BW
      KokkosBlas::gemm("T", "N", 1.0, X, AP, 0.0, XAP);  // XAP = X^T * AP
      KokkosBlas::gemm("T", "N", 1.0, X, BP, 0.0, XBP);  // XBP = X^T * BP
      KokkosBlas::gemm("T", "N", 1.0, W, AW, 0.0, WAW);  // WAW = W^T * AW
      KokkosBlas::gemm("T", "N", 1.0, W, BW, 0.0, WBW);  // WBW = W^T * BW
      KokkosBlas::gemm("T", "N", 1.0, W, AP, 0.0, WAP);  // WAP = W^T * AP
      KokkosBlas::gemm("T", "N", 1.0, W, BP, 0.0, WBP);  // WBP = W^T * BP
      KokkosBlas::gemm("T", "N", 1.0, P, AP, 0.0, PAP);  // PAP = P^T * AP
      KokkosBlas::gemm("T", "N", 1.0, P, BP, 0.0, PBP);  // PBP = P^T * BP

      Kokkos::parallel_for(
          "update_gram", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m, m}),
          KOKKOS_LAMBDA(const int i, const int j) {
            gramA(i + m, j) = XAW(j, i);
            gramB(i + m, j) = XBW(j, i);
            gramA(i + 2 * m, j) = XAP(j, i);
            gramB(i + 2 * m, j) = XBP(j, i);
            gramA(i + 2 * m, j + m) = WAP(j, i);
            gramB(i + 2 * m, j + m) = WBP(j, i);
          });
    }

    int mx = (k == 0) ? 2 * m : 3 * m;
    if (k == 0) {
      Kokkos::resize(gramA, 2 * m, 2 * m);
      Kokkos::resize(gramB, 2 * m, 2 * m);
      Kokkos::resize(v, m, 2 * m);
    }

    lapackage::sygvx<T>(gramA.data(), gramB.data(), mx, m, w.data(), v.data());

    if (k == 0) {
      Kokkos::resize(gramA, 3 * m, 3 * m);
      Kokkos::resize(gramB, 3 * m, 3 * m);
      Kokkos::resize(v, m, 3 * m);
    }

    /* Compute Ritz vectors */
    View2D<T> temp("X_AX_BX_0", 3 * n, m);

    auto S_AS_BS_X = Kokkos::subview(S_AS_BS, Kokkos::ALL(), Kokkos::make_pair(0, m));
    auto S_AS_BS_WP = Kokkos::subview(S_AS_BS, Kokkos::ALL(), Kokkos::make_pair(m, 3 * m));
    auto v_X = Kokkos::subview(v, Kokkos::ALL(), Kokkos::make_pair(0, m));
    auto v_WP = Kokkos::subview(v, Kokkos::ALL(), Kokkos::make_pair(m, 3 * m));

    KokkosBlas::gemm("N", "T", 1.0, S_AS_BS_WP, v_WP, 0.0, temp);  // temp = W * v(1) + P * v(2)
    Kokkos::deep_copy(P_AP_BP, temp);                              // P = temp
    KokkosBlas::gemm("N", "T", 1.0, S_AS_BS_X, v_X, 0.0, temp);    // temp = X * v(0)
    KokkosBlas::update(1.0, temp, 1.0, P_AP_BP, 0.0, X_AX_BX);     // X = temp + P * v(2)

    KokkosBlas::fill(norm2, 0.0);
    Kokkos::parallel_for(
        "R = AX - BX * w, R_ = AX + 2 * AX", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, m}),
        KOKKOS_LAMBDA(const int i, const int j) {
          if (is_convergent(j) == 0) {
            R(i, j) = AX(i, j) - BX(i, j) * w(j);
            if (j < m0) {
              Kokkos::atomic_add(&norm2(j, 0), R(i, j) * R(i, j));
              Kokkos::atomic_add(&norm2(j, 1), (R(i, j) + 2 * AX(i, j)) * (R(i, j) + 2 * AX(i, j)));
            }
          }
        });

    for (int i = 0; i < m0; i++) {
      if (is_convergent(i) == 0) {
        res(i) = sqrt(norm2(i, 0) / norm2(i, 1));  // res(i) = ||Ri|| / ||Ri_||
      }
    }

    /* Check convergence */
    bool is_converged =
        check_convergence(res.data(), is_convergent.data(), m0, k, maxiter, tol, verbose);

    if (is_converged) {
      break;
    }
  }

  /* Copy result back to wp, vp */
  View1D<T> w_result(wp, m0);
  View2D<T> v_result(vp, n, m0);
  auto X_m0 = Kokkos::subview(X, Kokkos::ALL(), Kokkos::make_pair(0, m0));
  auto w_m0 = Kokkos::subview(w, Kokkos::make_pair(0, m0));
  Kokkos::deep_copy(v_result, X_m0);
  Kokkos::deep_copy(w_result, w_m0);
}

template <typename T>
void lobpcgI_gpu(T* Ap, T* Bp, int n, int m, T* wp, T* vp, T* Xp = nullptr, T* Mp = nullptr,
                 double tol = 1e-8, int maxiter = 500, bool verbose = true) {
  // TODO: Bp is nullptr, use identity matrix
  // TODO: sygv for full matrix eigenvalue problem
  // TODO: if not converged, pick up the result for lowest residual
  // TODO: if the lowest residual is still above 1e-4, then fail
  // TODO: print residual precision based on tolerance set
  // TODO: Xp need to copy to device
  Xp = nullptr;
  const int m0 = m;                    // number of eigenpairs desired
  const int m1 = int(ceil(1.0 * m0));  // added number of eigenpairs to compute
  m = m0 + m1;                         // total number of eigenpairs to compute
  m = n < m ? n : m;                   // m cannot be larger than n

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

  /* Store in: [X W P | AX AW AP | BX BW BP] */
  View2D<T> S_AS_BS("vhstack: [X W P | AX AW AP | BX BW BP]", 3 * n, 3 * m);
  auto S = Kokkos::subview(S_AS_BS, Kokkos::make_pair(0, n), Kokkos::ALL());
  auto AS = Kokkos::subview(S_AS_BS, Kokkos::make_pair(n, 2 * n), Kokkos::ALL());
  auto BS = Kokkos::subview(S_AS_BS, Kokkos::make_pair(2 * n, 3 * n), Kokkos::ALL());

  auto X_AX_BX = Kokkos::subview(S_AS_BS, Kokkos::ALL(), Kokkos::make_pair(0, m));
  auto W_AW_BW = Kokkos::subview(S_AS_BS, Kokkos::ALL(), Kokkos::make_pair(m, 2 * m));
  auto P_AP_BP = Kokkos::subview(S_AS_BS, Kokkos::ALL(), Kokkos::make_pair(2 * m, 3 * m));

  auto X = Kokkos::subview(X_AX_BX, Kokkos::make_pair(0, n), Kokkos::ALL());
  auto W = Kokkos::subview(W_AW_BW, Kokkos::make_pair(0, n), Kokkos::ALL());
  auto P = Kokkos::subview(P_AP_BP, Kokkos::make_pair(0, n), Kokkos::ALL());

  auto AX = Kokkos::subview(X_AX_BX, Kokkos::make_pair(n, 2 * n), Kokkos::ALL());
  auto AW = Kokkos::subview(W_AW_BW, Kokkos::make_pair(n, 2 * n), Kokkos::ALL());
  auto AP = Kokkos::subview(P_AP_BP, Kokkos::make_pair(n, 2 * n), Kokkos::ALL());

  auto BX = Kokkos::subview(X_AX_BX, Kokkos::make_pair(2 * n, 3 * n), Kokkos::ALL());
  auto BW = Kokkos::subview(W_AW_BW, Kokkos::make_pair(2 * n, 3 * n), Kokkos::ALL());
  auto BP = Kokkos::subview(P_AP_BP, Kokkos::make_pair(2 * n, 3 * n), Kokkos::ALL());

  /* Compute XAX0 = X.T * A * X, XBX0 = X.T * B * X */
  Kokkos::DualView<T**> XAX0("XAX0", m, m);
  Kokkos::DualView<T**> XBX0("XBX0", m, m);
  if (Xp == nullptr) {
    Kokkos::parallel_for(
        "lobpcg::set: XAX0, XBX0",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {m, m}),
        KOKKOS_LAMBDA(int i, int j) {
          XAX0.d_view(i, j) = A(i, j);  // X = eye(n, m) -> XAX = A[:m, :m]
          XBX0.d_view(i, j) = B(i, j);  // X = eye(n, m) -> XBX = B[:m, :m]
        });
  } else {
    X = View2D<T>(Xp, n, m);
    KokkosBlas::gemm("N", "N", 1.0, A, X, 0.0, AX);            // AX = A * X
    KokkosBlas::gemm("N", "N", 1.0, B, X, 0.0, BX);            // BX = B * X
    KokkosBlas::gemm("T", "N", 1.0, X, AX, 0.0, XAX0.d_view);  // XAX0 = X.T * AX
    KokkosBlas::gemm("T", "N", 1.0, X, BX, 0.0, XBX0.d_view);  // XBX0 = X.T * BX
  }

  XAX0.modify_device();  // mark XAX as modified on device
  XBX0.modify_device();  // mark XBX as modified on device
  XAX0.sync_host();      // sync XAX from device to host
  XBX0.sync_host();      // sync XBX from device to host

  /* Solve generalized eigenvalue problem to get initial eigenvectors */
  Kokkos::DualView<T*> w("eigenvalues", m);
  Kokkos::DualView<T**> v("eigenvectors: column major", m, m);

  lapackage::sygvx<T>(XAX0.h_view.data(), XBX0.h_view.data(), m, m, w.h_view.data(),
                      v.h_view.data());

  w.modify_host();  // mark w as modified on host
  v.modify_host();  // mark v as modified on host
  w.sync_device();  // sync w from host to device
  v.sync_device();  // sync v from host to device

  /* Compute: X = X * v, AX = A * X * v, BX = B * X * v */
  if (Xp == nullptr) {
    auto A_nm = Kokkos::subview(A, Kokkos::ALL(), Kokkos::make_pair(0, m));
    auto B_nm = Kokkos::subview(B, Kokkos::ALL(), Kokkos::make_pair(0, m));
    auto X_nm = Kokkos::subview(X, Kokkos::make_pair(0, m), Kokkos::ALL());

    KokkosBlas::axpy(1.0, v.d_view, X_nm);                 // X = v
    KokkosBlas::gemm("N", "N", 1.0, A_nm, X_nm, 0.0, AX);  // AX = A * X
    KokkosBlas::gemm("N", "N", 1.0, B_nm, X_nm, 0.0, BX);  // BX = B * X
  } else {
    View2D<T> X0 = X;
    KokkosBlas::gemm("N", "N", 1.0, A, X, 0.0, AX);                    // AX = A * X
    KokkosBlas::gemm("N", "N", 1.0, B, X, 0.0, BX);                    // BX = B * X
    KokkosBlas::gemm("N", "N", 1.0, X_AX_BX, v.d_view, 0.0, X_AX_BX);  // X = X * v
  }

  View2D<T> M;
  if (Mp != nullptr) {
    M = View2D<T>(Mp, n, n);
  }

  /* Compute residual: R = AX - BX * w, A_norm, B_norm */
  View2D<T> R("R = AX - BX * w", n, m);
  Kokkos::parallel_for(
      "R = AX - BX * w", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n, m}),
      KOKKOS_LAMBDA(int i, int j) { R(i, j) = AX(i, j) - BX(i, j) * w.d_view(j); });

  /* Initial: norm array for [Xi, Wi, Pi] */
  View2D<T> norm("norm", m, 3);
  View2D<T> norm2("norm", m0, 2);
  KokkosBlas::fill(norm, 1.0);  // initialize norm to 1.0

  /* Initial: gram matrix for Rayleigh-Ritz */
  Kokkos::DualView<T**> gramA("gramA", 3 * m, 3 * m);
  Kokkos::DualView<T**> gramB("gramB", 3 * m, 3 * m);

  /* Initial: convergent array as all false: 0 in host */
  Kokkos::DualView<T*> is_convergent("convergent flag", m);
  Kokkos::DualView<T*> res("residual norm stored in host", m);

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

    /* Normalize: X, W, P, AX, AW, AP, BX, BW, BP */
    Kokkos::parallel_for(
        "Xi_norm", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, m}),
        KOKKOS_LAMBDA(const int i, const int j) {
          if (is_convergent.d_view(j) == 0) {
            Kokkos::atomic_add(&norm(j, 0), X(i, j) * X(i, j));
            Kokkos::atomic_add(&norm(j, 1), W(i, j) * W(i, j));
            Kokkos::atomic_add(&norm(j, 2), P(i, j) * P(i, j));
          }
        });

    Kokkos::parallel_for(
        3 * n, KOKKOS_LAMBDA(const int i) {
          for (int j = 0; j < m; ++j) {  // memory contiguity in row, i.e. along m
            if (is_convergent.d_view(j) == 0) {
              X_AX_BX(i, j) /= sqrt(norm(j, 0));
              W_AW_BW(i, j) /= sqrt(norm(j, 1));
              if (k != 0) {
                P_AP_BP(i, j) /= sqrt(norm(j, 2));
              }
            }
          }
        });

    /* Perform Rayleigh-Ritz procedure */
    if (m < 150) {
      KokkosBlas::gemm("T", "N", 1.0, S, AS, 0.0, gramA.d_view);  // gramA = S^T * AS
      KokkosBlas::gemm("T", "N", 1.0, S, BS, 0.0, gramB.d_view);  // gramB = S^T * BS
    } else {
      auto pair_0_m = Kokkos::make_pair(0, m);
      auto pair_m_2m = Kokkos::make_pair(m, 2 * m);
      auto pair_2m_3m = Kokkos::make_pair(2 * m, 3 * m);
      auto XAX = Kokkos::subview(gramA.d_view, pair_0_m, pair_0_m);
      auto XBX = Kokkos::subview(gramB.d_view, pair_0_m, pair_0_m);
      auto XAW = Kokkos::subview(gramA.d_view, pair_0_m, pair_m_2m);
      auto XBW = Kokkos::subview(gramB.d_view, pair_0_m, pair_m_2m);
      auto XAP = Kokkos::subview(gramA.d_view, pair_0_m, pair_2m_3m);
      auto XBP = Kokkos::subview(gramB.d_view, pair_0_m, pair_2m_3m);
      auto WAW = Kokkos::subview(gramA.d_view, pair_m_2m, pair_m_2m);
      auto WBW = Kokkos::subview(gramB.d_view, pair_m_2m, pair_m_2m);
      auto WAP = Kokkos::subview(gramA.d_view, pair_m_2m, pair_2m_3m);
      auto WBP = Kokkos::subview(gramB.d_view, pair_m_2m, pair_2m_3m);
      auto PAP = Kokkos::subview(gramA.d_view, pair_2m_3m, pair_2m_3m);
      auto PBP = Kokkos::subview(gramB.d_view, pair_2m_3m, pair_2m_3m);

      KokkosBlas::gemm("T", "N", 1.0, X, AX, 0.0, XAX);  // XAX = X^T * AX
      KokkosBlas::gemm("T", "N", 1.0, X, BX, 0.0, XBX);  // XBX = X^T * BX
      KokkosBlas::gemm("T", "N", 1.0, X, AW, 0.0, XAW);  // XAW = X^T * AW
      KokkosBlas::gemm("T", "N", 1.0, X, BW, 0.0, XBW);  // XBW = X^T * BW
      KokkosBlas::gemm("T", "N", 1.0, X, AP, 0.0, XAP);  // XAP = X^T * AP
      KokkosBlas::gemm("T", "N", 1.0, X, BP, 0.0, XBP);  // XBP = X^T * BP
      KokkosBlas::gemm("T", "N", 1.0, W, AW, 0.0, WAW);  // WAW = W^T * AW
      KokkosBlas::gemm("T", "N", 1.0, W, BW, 0.0, WBW);  // WBW = W^T * BW
      KokkosBlas::gemm("T", "N", 1.0, W, AP, 0.0, WAP);  // WAP = W^T * AP
      KokkosBlas::gemm("T", "N", 1.0, W, BP, 0.0, WBP);  // WBP = W^T * BP
      KokkosBlas::gemm("T", "N", 1.0, P, AP, 0.0, PAP);  // PAP = P^T * AP
      KokkosBlas::gemm("T", "N", 1.0, P, BP, 0.0, PBP);  // PBP = P^T * BP

      Kokkos::parallel_for(
          "update_gram", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m, m}),
          KOKKOS_LAMBDA(const int i, const int j) {
            gramA.d_view(i + m, j) = XAW(j, i);
            gramB.d_view(i + m, j) = XBW(j, i);
            gramA.d_view(i + 2 * m, j) = XAP(j, i);
            gramB.d_view(i + 2 * m, j) = XBP(j, i);
            gramA.d_view(i + 2 * m, j + m) = WAP(j, i);
            gramB.d_view(i + 2 * m, j + m) = WBP(j, i);
          });
    }

    gramA.modify_device();
    gramB.modify_device();
    gramA.sync_host();
    gramB.sync_host();

    int mx = (k == 0) ? 2 * m : 3 * m;
    if (k == 0) {
      Kokkos::resize(gramA.h_view, 2 * m, 2 * m);
      Kokkos::resize(gramB.h_view, 2 * m, 2 * m);
      Kokkos::resize(v.h_view, 2 * m, m);
      Kokkos::resize(v.d_view, 2 * m, m);
    }

    lapackage::sygvx<T>(gramA.h_view.data(), gramB.h_view.data(), mx, m, w.h_view.data(),
                        v.h_view.data());

    if (k == 0) {
      Kokkos::resize(gramA.h_view, 3 * m, 3 * m);
      Kokkos::resize(gramB.h_view, 3 * m, 3 * m);
      Kokkos::resize(v.h_view, 3 * m, m);
      Kokkos::resize(v.d_view, 3 * m, m);
    }

    w.modify_host();
    v.modify_host();
    w.sync_device();
    v.sync_device();

    /* Compute Ritz vectors */
    View2D<T> temp("X_AX_BX_0", 3 * n, m);

    auto S_AS_BS_X = Kokkos::subview(S_AS_BS, Kokkos::ALL(), Kokkos::make_pair(0, m));
    auto S_AS_BS_WP = Kokkos::subview(S_AS_BS, Kokkos::ALL(), Kokkos::make_pair(m, 3 * m));
    auto v_X = Kokkos::subview(v.d_view, Kokkos::make_pair(0, m), Kokkos::ALL());
    auto v_WP = Kokkos::subview(v.d_view, Kokkos::make_pair(m, 3 * m), Kokkos::ALL());

    KokkosBlas::gemm("N", "N", 1.0, S_AS_BS_WP, v_WP, 0.0, temp);  // temp = W * v(1) + P * v(2)
    Kokkos::deep_copy(P_AP_BP, temp);                              // P = temp
    KokkosBlas::gemm("N", "N", 1.0, S_AS_BS_X, v_X, 0.0, temp);    // temp = X * v(0)
    KokkosBlas::update(1.0, temp, 1.0, P_AP_BP, 0.0, X_AX_BX);     // X = temp + P * v(2)

    KokkosBlas::fill(norm2, 0.0);
    Kokkos::parallel_for(
        "R = AX - BX * w, R_ = AX + 2 * AX", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, m}),
        KOKKOS_LAMBDA(const int i, const int j) {
          if (is_convergent.d_view(j) == 0) {
            R(i, j) = AX(i, j) - BX(i, j) * w.d_view(j);
            if (j < m0) {
              Kokkos::atomic_add(&norm2(j, 0), R(i, j) * R(i, j));
              Kokkos::atomic_add(&norm2(j, 1), (R(i, j) + 2 * AX(i, j)) * (R(i, j) + 2 * AX(i, j)));
            }
          }
        });

    Kokkos::parallel_for(
        m0, KOKKOS_LAMBDA(const int i) {
          if (is_convergent.d_view(i) == 0) {
            res.d_view(i) = sqrt(norm2(i, 0) / norm2(i, 1));  // res(i) = ||Ri|| / ||Ri_||
          }
        });

    res.modify_device();
    res.sync_host();

    /* Check convergence */
    bool is_converged = check_convergence(res.h_view.data(), is_convergent.h_view.data(), m0, k,
                                          maxiter, tol, verbose);

    if (is_converged) {
      break;
    } else {
      is_convergent.modify_host();  // mark is_convergent as modified on host
      is_convergent.sync_device();  // sync is_convergent from host to device
    }
  }

  /* Copy result back to wp, vp */
  View1D<T> w_result(wp, m0);
  View2D<T> v_result(vp, n, m0);
  auto X_m0 = Kokkos::subview(X, Kokkos::ALL(), Kokkos::make_pair(0, m0));
  auto w_m0 = Kokkos::subview(w.d_view, Kokkos::make_pair(0, m0));
  Kokkos::deep_copy(v_result, X_m0);
  Kokkos::deep_copy(w_result, w_m0);
}

template <typename T>
void lobpcgII(T* Ap, T* Bp, int n, int m, T* wp, T* vp, T* Xp = nullptr, T* Mp = nullptr,
              double tol = 1e-8, int maxiter = 500, bool verbose = true) {
  // TODO: Bp is nullptr, use identity matrix
  // TODO: sygv for full matrix eigenvalue problem
  // TODO: if not converged, pick up the result for lowest residual
  // TODO: if the lowest residual is still above 1e-4, then fail
  // TODO: print residual precision based on tolerance set
  // TODO: Xp need to copy to device

  const int m0 = m;                    // number of eigenpairs desired
  const int m1 = int(ceil(1.0 * m0));  // added number of eigenpairs to compute
  m = m0 + m1;                         // total number of eigenpairs to compute
  m = n < m ? n : m;                   // m cannot be larger than n

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

  /* store in vstack [X, AX, BX, P, AP, BP, W, AW, BW] for host */
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

  /* Compute XAX = X.T * A * X, XBX = X.T * B * X */
  View2D<T> XAX("XAX", m, m);
  View2D<T> XBX("XBX", m, m);
  if (Xp == nullptr) {
    Kokkos::parallel_for(
        "lobpcg::set: XAX, XBX", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {m, m}),
        KOKKOS_LAMBDA(int i, int j) {
          XAX(i, j) = A(i, j);  // X = eye(n, m) -> XAX = A[:m, :m]
          XBX(i, j) = B(i, j);  // X = eye(n, m) -> XBX = B[:m, :m]
        });
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
    auto A_nm = Kokkos::subview(A, Kokkos::ALL(), Kokkos::make_pair(0, m));
    auto B_nm = Kokkos::subview(B, Kokkos::ALL(), Kokkos::make_pair(0, m));
    Kokkos::parallel_for(
        "X = eye(n, m) -> X = hstack(v, 0)",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {m, m}),
        KOKKOS_LAMBDA(int i, int j) { X(i, j) = v(j, i); });
    KokkosBlas::gemm("N", "T", 1.0, A_nm, v, 0.0, AX);  // AX = A * v
    KokkosBlas::gemm("N", "T", 1.0, B_nm, v, 0.0, BX);  // BX = B * v
  } else {
    View2D<T> X0 = X;
    KokkosBlas::gemm("N", "N", 1.0, A, X, 0.0, AX);             // AX = A * X
    KokkosBlas::gemm("N", "N", 1.0, B, X, 0.0, BX);             // BX = B * X
    KokkosBlas::gemm("N", "T", 1.0, X_AX_BX, v, 0.0, X_AX_BX);  // X = X * v
  }

  View2D<T> M;
  if (Mp != nullptr) {
    M = View2D<T>(Mp, n, n);
  }

  /* Compute residual: R = AX - BX * w, A_norm, B_norm */
  View1D<T> neg_one("all->1", m);
  View1D<T> one("all->1", m);
  KokkosBlas::fill(neg_one, -1.0);
  KokkosBlas::fill(one, 1.0);
  View2D<T> R("R = AX - BX * w", n, m);
  View2D<T> R_("R_ = AX + BX * w", n, m);
  Kokkos::deep_copy(R, AX);              // R = AX
  KokkosBlas::axpby(w, BX, neg_one, R);  // R = R - BX * w

  /* Initial norm array for [Xi, Wi, Pi] */
  View2D<T> norm("norm", m, 3);
  KokkosBlas::fill(norm, 1.0);
  for (int i = 0; i < m; i++) {
    norm(i, 0) = sqrt(KokkosBlas::nrm2_squared(Kokkos::subview(X, Kokkos::ALL(), i)));
    norm(i, 1) = sqrt(KokkosBlas::nrm2_squared(Kokkos::subview(R, Kokkos::ALL(), i)));
  }

  /* Initial for outer loop */
  View2D<T> gramAB_outer("vstack: [gramA_outer, gramB_outer]", 2 * m, m);
  auto gramA_outer = Kokkos::subview(gramAB_outer, Kokkos::make_pair(0, m), Kokkos::ALL());
  auto gramB_outer = Kokkos::subview(gramAB_outer, Kokkos::make_pair(m, 2 * m), Kokkos::ALL());
  View1D<T> w_outer("outer eigenvalues", m);
  View2D<T> v_outer("outer eigenvectors", m, m);

  /* Initial for inner loop */
  Kokkos::View<T[6][3]> gramAB_inner("vstack: [gramA_inner, gramB_inner]");
  auto gramA_inner = Kokkos::subview(gramAB_inner, Kokkos::make_pair(0, 3), Kokkos::ALL());
  auto gramB_inner = Kokkos::subview(gramAB_inner, Kokkos::make_pair(3, 6), Kokkos::ALL());
  Kokkos::View<T[1]> w_inner("inner eigenvalues");
  Kokkos::View<T[3]> v_inner("inner eigenvectors");

  View2D<T> S("vstack: [Xi, Wi, Pi], m sub-blocks[n, 3]", m * n, 3);
  View2D<T> ABS("vstack: [AXi, AWi, APi, BXi, BWi, BPi], m sub-blocks[n, 6]", m * n, 6);

  View1D<T> v0("v0", m);
  View1D<T> v1("v1", m);
  View1D<T> v2("v2", m);

  /* Initial convergent array as all false: 0 in host */
  View1D<T> is_convergent("convergent flag", m);
  View1D<T> res("residual norm stored in host", m);

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

    /* Perform Rayleigh-Ritz procedure */
    /* Normalize [Xi, Wi, Pi] and store them in contiguous memory for each sub-block */
    Kokkos::parallel_for(
        "vstack: S = [Xi, Wi, Pi], m sub-blocks[n, 3]"
        "vstack: ABS = [AXi, AWi, APi, BXi, BWi, BPi], m sub-blocks[n, 6]",
        Kokkos::RangePolicy<ExecSpace>(0, n), KOKKOS_LAMBDA(const int i) {
          for (int j = 0; j < m; ++j) {  // memory contiguity in row, i.e. along m
            if (is_convergent(j) == 0) {
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

    /* Perform inner Rayleigh-Ritz procedure */
    /* Use hard lock technique to lock the convergent eigenpairs */
    for (int i = 0; i < m; i++) {
      if (is_convergent(i) == 0) {  // only loop over non convergent eigenpairs
        /* Compute symmetric Gram matrices */
        auto Si = Kokkos::subview(S, Kokkos::make_pair(i * n, (i + 1) * n), Kokkos::ALL());
        auto ABSi = Kokkos::subview(ABS, Kokkos::make_pair(i * n, (i + 1) * n), Kokkos::ALL());
        KokkosBlas::gemm("T", "N", 1.0, ABSi, Si, 0.0, gramAB_inner);

        /* Make sure store gramA, gramB to contigous memory */
        if (k == 0) {
          gramA_inner(0, 2) = gramA_inner(1, 0);
          gramA_inner(1, 0) = gramA_inner(1, 1);
          gramB_inner(0, 2) = gramB_inner(1, 0);
          gramB_inner(1, 0) = gramB_inner(1, 1);
        }

        /* Compute eigenvalues and eigenvectors 3x3 eigenvalue problem */
        // int n_inner = (k == 0) ? 2 : 3;
        // lapackage::sygvx<T>(gramA_inner.data(), gramB_inner.data(), n_inner, 1,
        //                     w_inner.data(), v_inner.data());

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

    /* Compute the Ritz vector, compute batchly out of inner loop */
    KokkosBlas::axpby(v1, W_AW_BW, v2, P_AP_BP);   // P = W * v(1) + P * v(2)
    KokkosBlas::axpby(one, P_AP_BP, v0, X_AX_BX);  // X = X * v(0) + P

    /* Perform outer Rayleigh-Ritz procedure */
    KokkosBlas::gemm("T", "N", 1.0, X, AX, 0.0, gramA_outer);  // gramA = X^T * AX
    KokkosBlas::gemm("T", "N", 1.0, X, BX, 0.0, gramB_outer);  // gramB = X^T * BX

    /* Compute eigenvalues and eigenvectors for m x m eigenvalue problem */
    lapackage::sygvx<T>(gramA_outer.data(), gramB_outer.data(), m, m, w_outer.data(),
                        v_outer.data());

    /* [X, AX, BX, P, AP, BP] = [X, AX, BX, P, AP, BP] * v */
    Kokkos::deep_copy(ABXP0, ABXP);
    KokkosBlas::gemm("N", "T", 1.0, ABXP0, v_outer, 0.0, ABXP);

    /* Compute Residual: res = ||AX - BX * w|| / ||AX + BX * w|| */
    Kokkos::deep_copy(R, AX);                      // R = AX
    KokkosBlas::axpby(w_outer, BX, neg_one, R);    // R = BX * w - R
    KokkosBlas::update(1.0, R, 2.0, AX, 0.0, R_);  // R_ = R + 2*AX

    /* update residual norm, and norm of Xi, Wi, Pi */
    for (int i = 0; i < m; i++) {
      if (is_convergent(i) == 0) {
        auto Xi = Kokkos::subview(X, Kokkos::ALL(), i);
        auto Ri = Kokkos::subview(R, Kokkos::ALL(), i);
        auto Ri_ = Kokkos::subview(R_, Kokkos::ALL(), i);
        auto Pi = Kokkos::subview(P, Kokkos::ALL(), i);

        norm(i, 0) = sqrt(KokkosBlas::nrm2_squared(Xi));  // norm(i, 0) = ||Xi||
        norm(i, 1) = sqrt(KokkosBlas::nrm2_squared(Ri));  // norm(i, 1) = ||Ri||
        norm(i, 2) = sqrt(KokkosBlas::nrm2_squared(Pi));  // norm(i, 2) = ||Pi||

        if (i < m0) {
          T Ri_norm_ = sqrt(KokkosBlas::nrm2_squared(Ri_));  // ||Ri_||
          res(i) = norm(i, 1) / Ri_norm_;                    // res(i) = ||Ri|| / ||Ri_||
        }
      }
    }

    /* Check convergence */
    bool is_converged =
        check_convergence(res.data(), is_convergent.data(), m0, k, maxiter, tol, verbose);

    if (is_converged) {
      break;
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
void lobpcgII_gpu(T* Ap, T* Bp, int n, int m, T* wp, T* vp, T* Xp = nullptr, T* Mp = nullptr,
                  double tol = 1e-8, int maxiter = 500, bool verbose = true) {
  // TODO: Bp is nullptr, use identity matrix
  // TODO: sygv for full matrix eigenvalue problem
  // TODO: if not converged, pick up the result for lowest residual
  // TODO: if the lowest residual is still above 1e-4, then fail
  // TODO: print residual precision based on tolerance set
  // TODO: Xp need to copy to device

  const int m0 = m;                    // number of eigenpairs desired
  const int m1 = int(ceil(1.0 * m0));  // added number of eigenpairs to compute
  m = m0 + m1;                         // total number of eigenpairs to compute
  m = n < m ? n : m;                   // m cannot be larger than n

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

  /* store in hstack [X, AX, BX, P, AP, BP, W, AW, BW] for device */
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

  /* Compute XAX0 = X.T * A * X, XBX0 = X.T * B * X */
  Kokkos::DualView<T**> XAX0("XAX0", m, m);
  Kokkos::DualView<T**> XBX0("XBX0", m, m);

  if (Xp == nullptr) {
    Kokkos::parallel_for(
        "lobpcg::set: XAX0, XBX0",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {m, m}),
        KOKKOS_LAMBDA(int i, int j) {
          XAX0.d_view(i, j) = A(i, j);  // X = eye(n, m) -> XAX0 = A[:m, :m]
          XBX0.d_view(i, j) = B(i, j);  // X = eye(n, m) -> XBX0 = B[:m, :m]
        });
  } else {
    X = View2D<T>(Xp, n, m);
    KokkosBlas::gemm("N", "N", 1.0, A, X, 0.0, AX);            // AX = A * X
    KokkosBlas::gemm("N", "N", 1.0, B, X, 0.0, BX);            // BX = B * X
    KokkosBlas::gemm("T", "N", 1.0, X, AX, 0.0, XAX0.d_view);  // XAX0 = X.T * AX
    KokkosBlas::gemm("T", "N", 1.0, X, BX, 0.0, XBX0.d_view);  // XBX0 = X.T * BX
  }
  XAX0.modify_device();  // mark XAX0 as modified on device
  XBX0.modify_device();  // mark XBX0 as modified on device
  XAX0.sync_host();      // sync XAX0 from device to host
  XBX0.sync_host();      // sync XBX0 from device to host

  /* Solve generalized eigenvalue problem to get initial eigenvectors */
  Kokkos::DualView<T*> w("eigenvalues", m);
  Kokkos::DualView<T**> v("eigenvectors column major", m, m);

  lapackage::sygvx<T>(XAX0.h_view.data(), XBX0.h_view.data(), m, m, w.h_view.data(),
                      v.h_view.data());

  w.modify_host();  // mark w as modified on host
  v.modify_host();  // mark v as modified on host
  w.sync_device();  // sync w from host to device
  v.sync_device();  // sync v from host to device

  /* Compute: X = X * v, AX = A * X * v, BX = B * X * v */
  if (Xp == nullptr) {
    auto A_nm = Kokkos::subview(A, Kokkos::ALL(), Kokkos::make_pair(0, m));
    auto B_nm = Kokkos::subview(B, Kokkos::ALL(), Kokkos::make_pair(0, m));
    auto X_nm = Kokkos::subview(X, Kokkos::make_pair(0, m), Kokkos::ALL());

    KokkosBlas::axpy(1.0, v.d_view, X_nm);                 // X = v
    KokkosBlas::gemm("N", "N", 1.0, A_nm, X_nm, 0.0, AX);  // AX = A * X
    KokkosBlas::gemm("N", "N", 1.0, B_nm, X_nm, 0.0, BX);  // BX = B * X
  } else {
    View2D<T> X0 = X;
    KokkosBlas::gemm("N", "N", 1.0, A, X, 0.0, AX);                    // AX = A * X
    KokkosBlas::gemm("N", "N", 1.0, B, X, 0.0, BX);                    // BX = B * X
    KokkosBlas::gemm("N", "N", 1.0, X_AX_BX, v.d_view, 0.0, X_AX_BX);  // X = X * v
  }

  View2D<T> M;
  if (Mp != nullptr) {
    M = View2D<T>(Mp, n, n);
  }

  /* Compute residual: R = AX - BX * w, A_norm, B_norm */
  View1D<T> neg_one("all->1", m);
  View1D<T> one("all->1", m);
  KokkosBlas::fill(neg_one, -1.0);
  KokkosBlas::fill(one, 1.0);
  View2D<T> R("R = AX - BX * w", n, m);
  View2D<T> R_("R_ = AX + BX * w", n, m);
  Kokkos::deep_copy(R, AX);                     // R = AX
  KokkosBlas::axpby(w.d_view, BX, neg_one, R);  // R = R - BX * w

  /* Initial norm array for [Xi, Wi, Pi] */
  View2D<T> norm("norm", m, 3);
  KokkosBlas::fill(norm, 1.0);
  for (int i = 0; i < m; i++) {
    T Xi_norm = sqrt(KokkosBlas::nrm2_squared(Kokkos::subview(X, Kokkos::ALL(), i)));
    T Wi_norm = sqrt(KokkosBlas::nrm2_squared(Kokkos::subview(R, Kokkos::ALL(), i)));
    Kokkos::deep_copy(Kokkos::subview(norm, i, 0), Xi_norm);
    Kokkos::deep_copy(Kokkos::subview(norm, i, 1), Wi_norm);
  }

  /* Initial for outer loop */
  Kokkos::DualView<T**> gramAB_outer("hstack: [gramA_outer, gramB_outer]", m, 2 * m);
  auto gramA_outer = Kokkos::subview(gramAB_outer, Kokkos::ALL(), Kokkos::make_pair(0, m));
  auto gramB_outer = Kokkos::subview(gramAB_outer, Kokkos::ALL(), Kokkos::make_pair(m, 2 * m));
  Kokkos::DualView<T*> w_outer("outer eigenvalues", m);
  Kokkos::DualView<T**> v_outer("outer eigenvectors", m, m);

  /* Initial for inner loop */
  Kokkos::DualView<T**> gramAB_inner("hstack: [gramA_inner, gramB_inner]", 3, 6);
  auto gramA_inner = Kokkos::subview(gramAB_inner, Kokkos::ALL(), Kokkos::make_pair(0, 3));
  auto gramB_inner = Kokkos::subview(gramAB_inner, Kokkos::ALL(), Kokkos::make_pair(3, 6));
  Kokkos::DualView<T*> w_inner("inner eigenvalues", 1);
  Kokkos::DualView<T*> v_inner("inner eigenvectors", 3);

  View2D<T> S("hstack: [Xi, Wi, Pi], m sub-blocks[n, 3]", n, 3 * m);
  View2D<T> ABS("hstack: [AXi, AWi, APi, BXi, BWi, BPi], m sub-blocks[n, 6]", n, 6 * m);

  Kokkos::DualView<T*> v0("v0", m);
  Kokkos::DualView<T*> v1("v1", m);
  Kokkos::DualView<T*> v2("v2", m);

  /* Initial convergent array as all false: 0 in host */
  Kokkos::DualView<T*> is_convergent("convergent flag", m);
  Kokkos::View<T*, Kokkos::HostSpace> res("residual norm stored in host", m);

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

    /* Perform Rayleigh-Ritz procedure */
    /* Normalize [Xi, Wi, Pi] and store them in contiguous memory for each sub-block */
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

    /* Perform inner Rayleigh-Ritz procedure */
    /* Use hard lock technique to lock the convergent eigenpairs */
    for (int i = 0; i < m; i++) {
      if (is_convergent.h_view(i) == 0) {  // only loop over non convergent eigenpairs
        /* Compute symmetric Gram matrices */
        auto Si = Kokkos::subview(S, Kokkos::ALL(), Kokkos::make_pair(i * 3, i * 3 + 3));
        auto ABSi = Kokkos::subview(ABS, Kokkos::ALL(), Kokkos::make_pair(i * 6, i * 6 + 6));
        KokkosBlas::gemm("T", "N", 1.0, Si, ABSi, 0.0, gramAB_inner.d_view);

        gramAB_inner.modify_device();
        gramAB_inner.sync_host();

        /* Make sure store gramA, gramB to contigous memory */
        // tick("inner loop 0");
        if (k == 0) {
          gramA_inner.h_view(2, 0) = gramA_inner.h_view(0, 1);
          gramB_inner.h_view(2, 0) = gramB_inner.h_view(0, 1);
          gramA_inner.h_view(0, 1) = gramA_inner.h_view(1, 1);
          gramB_inner.h_view(0, 1) = gramB_inner.h_view(1, 1);
        }

        /* Compute eigenvalues and eigenvectors 3x3 eigenvalue problem */
        // int n_inner = (k == 0) ? 2 : 3;
        // lapackage::sygvx<T>(gramA_inner.h_view.data(), gramB_inner.h_view.data(), n_inner, 1,
        //                     w_inner.h_view.data(), v_inner.h_view.data());

        if (k == 0) {
          sygvx2x2(gramA_inner.h_view.data(), gramB_inner.h_view.data(), 1, w_inner.h_view.data(),
                   v_inner.h_view.data());
        } else {
          sygvx3x3(gramA_inner.h_view.data(), gramB_inner.h_view.data(), 1, w_inner.h_view.data(),
                   v_inner.h_view.data());
        }

        /* Only store the result, move the computation out of the loop */
        v0.h_view(i) = v_inner.h_view(0);
        v1.h_view(i) = v_inner.h_view(1);
        v2.h_view(i) = v_inner.h_view(2);
      }
    }

    v0.modify_host();
    v1.modify_host();
    v2.modify_host();
    v0.sync_device();
    v1.sync_device();
    v2.sync_device();

    /* Compute the Ritz vector, compute batchly out of inner loop */
    View1D<T> one("all->1", 3 * m);
    Kokkos::deep_copy(one, 1.0);
    View1D<T> v0_v0_v0("v0_v0_v0", 3 * m);
    View1D<T> v1_v1_v1("v1_v1_v1", 3 * m);
    View1D<T> v2_v2_v2("v2_v2_v2", 3 * m);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecSpace>(0, m), KOKKOS_LAMBDA(const int i) {
          v0_v0_v0(i) = v0.d_view(i);
          v1_v1_v1(i) = v1.d_view(i);
          v2_v2_v2(i) = v2.d_view(i);

          v0_v0_v0(i + m) = v0.d_view(i);
          v1_v1_v1(i + m) = v1.d_view(i);
          v2_v2_v2(i + m) = v2.d_view(i);

          v0_v0_v0(i + 2 * m) = v0.d_view(i);
          v1_v1_v1(i + 2 * m) = v1.d_view(i);
          v2_v2_v2(i + 2 * m) = v2.d_view(i);
        });

    KokkosBlas::axpby(v1_v1_v1, W_AW_BW, v2_v2_v2, P_AP_BP);  // P = W * v(1) + P * v(2)
    KokkosBlas::axpby(one, P_AP_BP, v0_v0_v0, X_AX_BX);       // X = X * v(0) + P

    /* Perform outer Rayleigh-Ritz procedure */
    KokkosBlas::gemm("T", "N", 1.0, X, AX, 0.0, gramA_outer.d_view);  // gramA = X^T * AX
    KokkosBlas::gemm("T", "N", 1.0, X, BX, 0.0, gramB_outer.d_view);  // gramB = X^T * BX

    /* Compute eigenvalues and eigenvectors for m x m eigenvalue problem */
    gramAB_outer.modify_device();
    gramAB_outer.sync_host();

    lapackage::sygvx<T>(gramA_outer.h_view.data(), gramB_outer.h_view.data(), m, m,
                        w_outer.h_view.data(), v_outer.h_view.data());

    w_outer.modify_host();
    v_outer.modify_host();
    w_outer.sync_device();
    v_outer.sync_device();

    /* [X, AX, BX, P, AP, BP] = [X, AX, BX, P, AP, BP] * v */
    Kokkos::deep_copy(ABXP0, ABXP);

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

    /* Compute Residual: res = ||AX - BX * w|| / ||AX + BX * w|| */
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
        T Xi_norm = sqrt(KokkosBlas::nrm2_squared(Kokkos::subview(X, Kokkos::ALL(), i)));
        T Wi_norm = sqrt(KokkosBlas::nrm2_squared(Kokkos::subview(R, Kokkos::ALL(), i)));
        T Pi_norm = sqrt(KokkosBlas::nrm2_squared(Kokkos::subview(P, Kokkos::ALL(), i)));
        Kokkos::deep_copy(Kokkos::subview(norm, i, 0), Xi_norm);
        Kokkos::deep_copy(Kokkos::subview(norm, i, 1), Wi_norm);
        Kokkos::deep_copy(Kokkos::subview(norm, i, 2), Pi_norm);
      }
    }

    /* Check convergence */
    bool is_converged =
        check_convergence(res.data(), is_convergent.h_view.data(), m0, k, maxiter, tol, verbose);

    if (is_converged) {
      break;
    } else {
      is_convergent.modify_host();  // mark is_convergent as modified on host
      is_convergent.sync_device();  // sync is_convergent from host to device
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
void lobpcg(T* Ap, T* Bp, int n, int m, T* wp, T* vp, T* Xp = nullptr, T* Mp = nullptr,
            double tol = 1e-8, int maxiter = 500, bool verbose = true) {
#ifdef KOKKOS_ENABLE_CUDA
  lobpcgII_gpu(Ap, Bp, n, m, wp, vp, Xp, Mp, tol, maxiter, verbose);
  // TODO: gpu m1 more are better, since need less data transfer
#else
  lobpcgII(Ap, Bp, n, m, wp, vp, Xp, Mp, tol, maxiter, verbose);
#endif
}

}  // namespace linalg

#endif  // LOBPCG_HPP