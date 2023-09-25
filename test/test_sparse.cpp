#include <gtest/gtest.h>

#include <KokkosBlas1_scal.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <Kokkos_Random.hpp>
#include <array>

#include "lapackage.hpp"
#include "sparse.hpp"
#include "utils.hpp"

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

typedef int I;
typedef double T;

T Ax0[7] = {1, 1, 1, 1, 1, 1, 1};
I Ai0[7] = {3, 0, 0, 1, 1, 0, 0};
I Aj0[7] = {3, 2, 0, 1, 1, 0, 0};

T Bx_ans[4] = {3, 1, 2, 1};
I Bp_ans[5] = {0, 2, 3, 3, 4};
I Bj_ans[4] = {0, 2, 1, 3};

TEST(CooToCsrTest, coo_to_csr) {
  std::vector<T> Ax(Ax0, Ax0 + 7);
  std::vector<I> Ai(Ai0, Ai0 + 7);
  std::vector<I> Aj(Aj0, Aj0 + 7);

  std::vector<T> Bx;
  std::vector<I> Bp;
  std::vector<I> Bj;

  std::tie(Bx, Bp, Bj) = coo_to_csr<T, I>(Ax, Ai, Aj);

  printmat("Bx", Bx.data(), 1, 4);
  printmat("Bp", Bp.data(), 1, 5);
  printmat("Bj", Bj.data(), 1, 4);

  ASSERT_EQ(Bx.size(), 4);
  ASSERT_EQ(Bp.size(), 5);
  ASSERT_EQ(Bj.size(), 4);

  for (size_t i = 0; i < Bx.size(); ++i) {
    ASSERT_EQ(Bx[i], Bx_ans[i]);
  }

  for (size_t i = 0; i < Bp.size(); ++i) {
    ASSERT_EQ(Bp[i], Bp_ans[i]);
  }

  for (size_t i = 0; i < Bj.size(); ++i) {
    ASSERT_EQ(Bj[i], Bj_ans[i]);
  }
}

TEST(CooToCsrTest, cooToCsr) {
  COOMatrix<T, I> coo;
  coo.rowIndices = Kokkos::View<int*>(&Ai0[0], std::size(Ai0));
  coo.colIndices = Kokkos::View<int*>(&Aj0[0], std::size(Aj0));
  coo.values = Kokkos::View<double*>(&Ax0[0], std::size(Ax0));

  // Convert COO to CSR
  CSRMatrix<T, I> csr;
  cooToCsr(coo, csr);

  // Check the results
  for (size_t i = 0; i < std::size(Bx_ans); ++i) {
    ASSERT_EQ(Bx_ans[i], csr.values(i));
  }

  for (size_t i = 0; i < std::size(Bp_ans); ++i) {
    ASSERT_EQ(Bp_ans[i], csr.rowPtr(i));
  }

  for (size_t i = 0; i < std::size(Bj_ans); ++i) {
    ASSERT_EQ(Bj_ans[i], csr.colIndices(i));
  }
}

TEST(SpeedTest, coo_to_csr) {
  // random generated sparse matrix with size 1000000 x 1000000
  // with 10000000 non-zero elements
  I n = 640000;
  I nnz = 20403264;

  std::vector<T> Ax(nnz);
  std::vector<I> Ai(nnz);
  std::vector<I> Aj(nnz);

  // generate random sparse matrix
  for (size_t i = 0; i < nnz; ++i) {
    Ax[i] = 1;
    Ai[i] = rand() % n;
    Aj[i] = rand() % n;
  }

  std::vector<T> Bx;
  std::vector<I> Bp;
  std::vector<I> Bj;

  tick("std::vector");
  std::tie(Bx, Bp, Bj) = coo_to_csr<T, I>(Ax, Ai, Aj, n);
  tock("std::vector");
}

TEST(SpeedTest, cooToCsr) {
  // random generated sparse matrix with size 1000000 x 1000000
  // with 10000000 non-zero elements
  I n = 640000;
  I nnz = 20403264;

  std::vector<T> Ax(nnz);
  std::vector<I> Ai(nnz);
  std::vector<I> Aj(nnz);

  // generate random sparse matrix
  for (size_t i = 0; i < nnz; ++i) {
    Ax[i] = 1;
    Ai[i] = rand() % n;
    Aj[i] = rand() % n;
  }

  COOMatrix<T, I> coo;
  coo.numRows = n;
  coo.rowIndices = Kokkos::View<int*>(&Ai[0], std::size(Ai));
  coo.colIndices = Kokkos::View<int*>(&Aj[0], std::size(Aj));
  coo.values = Kokkos::View<double*>(&Ax[0], std::size(Ax));

  CSRMatrix<T, I> csr;

  tick("Kokkos::View");
  cooToCsr(coo, csr);
  tock("Kokkos::View");
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);

  ::testing::InitGoogleTest(&argc, argv);

  int result = RUN_ALL_TESTS();

  Kokkos::finalize();
  return result;
}
