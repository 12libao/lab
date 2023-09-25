#include <gtest/gtest.h>

#include <KokkosBlas.hpp>
#include <KokkosSparse_gmres.hpp>
#include <Kokkos_Random.hpp>
#include <cstdlib>

#include "KokkosSparse_IOUtils.hpp"
#include "KokkosSparse_spgemm.hpp"
#include "lobpcg.hpp"
#include "lobpcgSparse.hpp"
#include "sparse.hpp"
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

typedef int I;
typedef double T;

class FixSysPosMat : public ::testing::Test {
 public:
  T Ax_data[100] = {
      4.61107843,  0.55522023,  -0.47604613, 0.66740675,  -0.08236241, 0.77442043,  -0.16847107,
      0.08554566,  -0.1860363,  -0.89421982, 0.55522023,  7.57783175,  -0.96999787, 0.41899955,
      -0.74169562, -0.83982537, -0.16069949, 0.89837447,  -0.12617254, 0.33375998,  -0.47604613,
      -0.96999787, 6.25321263,  -1.04613187, 0.48408668,  -0.24834547, -0.9308837,  0.55077296,
      0.10224118,  -0.79222648, 0.66740675,  0.41899955,  -1.04613187, 6.48985449,  -0.21958628,
      0.15202723,  1.15379689,  0.98174814,  0.75593,     -0.81617607, -0.08236241, -0.74169562,
      0.48408668,  -0.21958628, 7.34816783,  0.46234805,  -0.13316185, -0.45256164, -0.23745895,
      -0.11623706, 0.77442043,  -0.83982537, -0.24834547, 0.15202723,  0.46234805,  6.69636965,
      -0.58163821, 0.43759353,  0.05546983,  -0.4466029,  -0.16847107, -0.16069949, -0.9308837,
      1.15379689,  -0.13316185, -0.58163821, 5.23492206,  0.04704558,  -0.47210199, 0.33053751,
      0.08554566,  0.89837447,  0.55077296,  0.98174814,  -0.45256164, 0.43759353,  0.04704558,
      6.63828556,  -0.03473362, 0.05309124,  -0.1860363,  -0.12617254, 0.10224118,  0.75593,
      -0.23745895, 0.05546983,  -0.47210199, -0.03473362, 8.48108133,  -1.02463494, -0.89421982,
      0.33375998,  -0.79222648, -0.81617607, -0.11623706, -0.4466029,  0.33053751,  0.05309124,
      -1.02463494, 7.15113258};

  I Aj_data[100] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
                    5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
                    5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  I Ap_data[11] = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

  T Bx_data[100] = {
      1.40247602e+00,  9.71640121e-02,  -9.33275114e-02, -7.08290225e-02, 3.75605730e-04,
      -1.58986854e-01, 4.69279444e-02,  -1.50059034e-01, 1.59782560e-01,  9.23926265e-02,
      9.71640121e-02,  1.63109948e+00,  3.97920065e-02,  5.52071516e-02,  -6.60095576e-02,
      -2.08674272e-02, 7.29017308e-02,  -1.11165631e-03, 3.95364128e-02,  -2.26175451e-01,
      -9.33275114e-02, 3.97920065e-02,  1.51830773e+00,  9.84590157e-02,  7.29508523e-02,
      -5.51455823e-02, -2.53701148e-02, -8.70912341e-02, -4.21334552e-03, -5.38162686e-02,
      -7.08290225e-02, 5.52071516e-02,  9.84590157e-02,  1.58193477e+00,  -1.59398144e-01,
      -5.06519711e-03, -9.63670377e-03, -2.16515458e-01, -3.26136823e-02, -1.89053463e-02,
      3.75605730e-04,  -6.60095576e-02, 7.29508523e-02,  -1.59398144e-01, 1.65732476e+00,
      -9.63299199e-02, 1.07806411e-01,  9.54309750e-02,  2.00620402e-02,  -1.88117418e-02,
      -1.58986854e-01, -2.08674272e-02, -5.51455823e-02, -5.06519711e-03, -9.63299199e-02,
      1.51002727e+00,  1.43406417e-02,  -1.51418907e-01, -1.58634934e-01, 7.17075180e-03,
      4.69279444e-02,  7.29017308e-02,  -2.53701148e-02, -9.63670377e-03, 1.07806411e-01,
      1.43406417e-02,  1.33844572e+00,  -1.01970936e-01, -1.91239978e-02, -2.87536341e-02,
      -1.50059034e-01, -1.11165631e-03, -8.70912341e-02, -2.16515458e-01, 9.54309750e-02,
      -1.51418907e-01, -1.01970936e-01, 1.50181428e+00,  -3.84895420e-02, 7.32017571e-02,
      1.59782560e-01,  3.95364128e-02,  -4.21334552e-03, -3.26136823e-02, 2.00620402e-02,
      -1.58634934e-01, -1.91239978e-02, -3.84895420e-02, 1.42023128e+00,  1.02036842e-01,
      9.23926265e-02,  -2.26175451e-01, -5.38162686e-02, -1.89053463e-02, -1.88117418e-02,
      7.17075180e-03,  -2.87536341e-02, 7.32017571e-02,  1.02036842e-01,  1.47385147e+00};

  I Bj_data[100] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
                    5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
                    5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  I Bp_data[11] = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

  T w_data[10] = {2.18001266, 2.51990827, 2.75329708, 3.90893235, 4.2813551,
                  4.9965097,  5.0351763,  5.94017474, 6.54900011, 8.35773};

  T v_data[100] = {
      -0.36568376, 0.53458882,  -0.08425405, 0.11737942,  -0.26585255, -0.007811,   -0.00847441,
      0.36821745,  -0.19776121, 0.25947002,  0.16509909,  0.06659396,  0.28786448,  -0.17570303,
      -0.26259852, -0.49381132, -0.06231013, -0.15836076, -0.3098011,  -0.22694697, 0.26315862,
      0.35801608,  0.08802997,  -0.27639771, 0.24018532,  -0.12177436, 0.45021287,  0.23019121,
      0.23058159,  0.0886961,   0.36555391,  0.11324762,  -0.3575999,  -0.19256977, 0.04200757,
      0.19413537,  -0.20088417, -0.10443037, -0.40399963, 0.30199261,  -0.12723799, -0.01985568,
      0.09358583,  -0.43921288, 0.18505825,  -0.06181628, -0.55489877, 0.21373643,  0.09313537,
      0.03035562,  0.24401814,  -0.0509544,  0.22930261,  0.43029066,  0.20311277,  -0.26908234,
      -0.19532623, 0.29543772,  -0.05026158, 0.38583235,  -0.04478387, 0.18512659,  0.56313051,
      0.05254134,  0.25405063,  0.47731877,  0.06861497,  -0.19096077, -0.26317057, -0.00881083,
      -0.29163712, -0.26587136, -0.02830397, -0.15118387, 0.25446925,  -0.14720062, 0.33181109,
      0.08759725,  -0.44577622, 0.35514387,  -0.1255203,  0.13089124,  0.03983545,  0.03173073,
      0.12574469,  -0.21664864, -0.07150045, -0.58656696, 0.26979744,  0.45266803,  -0.03631802,
      0.26115449,  -0.21292441, 0.16526491,  0.4793798,   -0.24723653, -0.09602952, -0.09375213,
      -0.18601902, -0.46715739};

  I n = 10;
  I m = 2;
};

TEST_F(FixSysPosMat, testsparse) {
  View2D<T> A0(&Ax_data[0], n, n);
  View2D<T> B0(&Bx_data[0], n, n);
  View1D<T> w(&w_data[0], n);

  // compute Aj and Ap for n x m dense matrix
  m = 10;
  View1D<I> Aj("Aj", n * m);
  View1D<I> Ap("Ap", n + 1);
  for (size_t i = 0; i < n; ++i) {
    Ap(i) = i * m;
    for (size_t j = 0; j < m; ++j) {
      Aj(i * m + j) = j;
    }
  }
  Ap(n) = n * m;

  View1D<I> Bp = Ap;
  View1D<I> Bj = Aj;

  // make kernel handle and set the options for GMRES
  using EXSP = Kokkos::DefaultExecutionSpace;
  using MESP = typename EXSP::memory_space;
  using crsMat_t = KokkosSparse::CrsMatrix<T, I, EXSP, void, I>;
  using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<I, I, T, EXSP, MESP, MESP>;

  crsMat_t A("A", n, n, Ap(n), nullptr, Ap.data(), Aj.data());
  crsMat_t B("B", n, n, Bp(n), &Bx_data[0], Bp.data(), Bj.data());

  // allocate the values for A
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      A.values(i * m + j) = A0(i, j);
    }
  }

  // check the matrix
  ASSERT_EQ(A.numRows(), n);
  ASSERT_EQ(A.numCols(), n);
  ASSERT_EQ(A.nnz(), Ap_data[n]);

  // printmat("A", A.values.data(), 10, 10);
  // printmat("Ap", A.graph.row_map.data(), 1, 11);
  // printmat("Aj", A.graph.entries.data(), 10, 10);

  // compute A * w
  View1D<T> Aw("Aw", n);
  View1D<T> Aw2("Aw", n);

  KokkosSparse::spmv("N", 1.0, A, w, 0.0, Aw);
  KokkosBlas::gemv("N", 1.0, A0, w, 0.0, Aw2);

  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(Aw(i), Aw2(i), 1e-8);
  }

  KernelHandle kh;
  kh.set_team_work_size(16);
  kh.set_dynamic_scheduling(true);
  std::string myalg("SPGEMM_KK_MEMORY");
  kh.create_spgemm_handle(KokkosSparse::StringToSPGEMMAlgorithm(myalg));

  // compute A * B
  crsMat_t C;
  KokkosSparse::spgemm_symbolic(kh, A, 0, B, 0, C);
  KokkosSparse::spgemm_numeric(kh, A, 0, B, 0, C);

  Kokkos::View<T**> C0("Cw", n, n);
  KokkosBlas::gemm("N", "N", 1.0, A0, B0, 0.0, C0);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      EXPECT_NEAR(C.values(i * n + j), C0(i, j), 1e-8);
    }
  }

  n = 1e6;
  crsMat_t A1 = KokkosSparse::Impl::kk_generate_diag_matrix<crsMat_t>(n);
  crsMat_t B1 = KokkosSparse::Impl::kk_generate_diag_matrix<crsMat_t>(n);
  crsMat_t C1;

  std::string alg1("SPGEMM_KK_MEMORY");
  kh.create_spgemm_handle(KokkosSparse::StringToSPGEMMAlgorithm(alg1));

  tick("SPGEMM_KK_MEMORY");
  KokkosSparse::spgemm_symbolic(kh, A1, 0, B1, 0, C1);
  KokkosSparse::spgemm_numeric(kh, A1, 0, B1, 0, C1);
  tock("SPGEMM_KK_MEMORY");

  std::string alg2("SPGEMM_KK_MEMSPEED");
  kh.create_spgemm_handle(KokkosSparse::StringToSPGEMMAlgorithm(alg2));

  tick("SPGEMM_KK_MEMSPEED");
  KokkosSparse::spgemm_symbolic(kh, A1, 0, B1, 0, C1);
  KokkosSparse::spgemm_numeric(kh, A1, 0, B1, 0, C1);
  tock("SPGEMM_KK_MEMSPEED");
}

TEST_F(FixSysPosMat, lobpcg) {
  tick("linalg::lobpcg");
  linalg::lobpcg<T>(&Ax_data[0], &Bx_data[0], n, m, &w_data[0], &v_data[0]);
  tock("linalg::lobpcg");

  tick("linalg::lobpcgsparse");
  linalg::sparse::lobpcg<T, I>(&Ax_data[0], &Ap_data[0], &Aj_data[0], &Bx_data[0], &Bp_data[0],
                               &Bj_data[0], n, m, &w_data[0], &v_data[0]);
  tock("linalg::lobpcgsparse");
}

TEST(SpeedTest, testgmres) {
  // // generate random matrix
  I n = 1e3;

  // make kernel handle and set the options for GMRES
  using EXSP = Kokkos::DefaultExecutionSpace;
  using MESP = typename EXSP::memory_space;
  using crsMat_t = KokkosSparse::CrsMatrix<T, I, EXSP, void, I>;
  using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<I, I, T, EXSP, MESP, MESP>;

  int m = 100;            // Max subspace size before restarting.
  double convTol = 1e-8;  // Relative residual convergence tolerance.
  int cycLim = 100;       // Maximum number of times to restart the solver.

  KernelHandle kh;
  kh.create_gmres_handle(m, convTol, cycLim);

  crsMat_t A = KokkosSparse::Impl::kk_generate_diag_matrix<crsMat_t>(n);
  View1D<T> x("x", n);
  View1D<T> b("b", n);
  int rand_seed = 123;
  Kokkos::Random_XorShift64_Pool<> pool(rand_seed);
  Kokkos::fill_random(b, pool, -1, 1);

  // use GMRES solver
  Kokkos::Timer timer;
  KokkosSparse::Experimental::gmres(&kh, A, b, x);
  double solve_time = timer.seconds();
  timer.reset();

  int printInfo = 1;
  if (printInfo) {
    const auto numIters = kh.get_gmres_handle()->get_num_iters();
    const auto convFlag = kh.get_gmres_handle()->get_conv_flag_val();
    const auto endRelRes = kh.get_gmres_handle()->get_end_rel_res();
    {
      printf("-------------------------------------------------------\n");
      printf("%-35s\n", (convFlag == 0) ? "GMRES: converged! :D" : "GMRES: not converged :(");
      printf("%-35s%10s\033[32m%10d\033[0m\n", "    Matrix size n:", "", n);
      printf("%-35s%10s\033[32m%10d\033[0m\n", "    Number of nonzeros nnz:", "", A.nnz());
      printf("%-35s%10s\033[32m%10d\033[0m\n", "    Total Number of Iterations:", "", numIters);
      printf("%-35s%8s\033[32m%10.6e\033[0m\n", "    Total Reduction in Residual:", "", endRelRes);
      printf("%-35s%8s\033[32m%10.6f s\033[0m\n", "    Solve time:", "", solve_time);
      printf("-------------------------------------------------------\n");
    }
  }
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);

  ::testing::InitGoogleTest(&argc, argv);

  int result = RUN_ALL_TESTS();

  Kokkos::finalize();

  return result;
}
