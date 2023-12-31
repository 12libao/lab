#include <gtest/gtest.h>

#include <KokkosBlas.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_Random.hpp>
#include <array>
#include <cstdlib>

#include "lobpcg.hpp"
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

constexpr I n = 10000;  // dof = 100,000,000 (one billion)
constexpr I m = 30;     // num eigenvalues

int maxiter = 100;
double tol = 1e-6;

class RandSysPosMat : public ::testing::Test {
 public:
  virtual void SetUp() override {
    A = Kokkos::DualView<T**>("A", n, n);
    B = Kokkos::DualView<T**>("B", n, n);

    auto h_A = A.h_view;
    auto h_B = B.h_view;

    auto d_A = A.d_view;
    auto d_B = B.d_view;

    // fill A and B with random numbers
    Kokkos::Random_XorShift64_Pool<ExecSpace> rand_pool(1234);
    Kokkos::fill_random(A.d_view, rand_pool, 0.0, 1.0);
    Kokkos::fill_random(B.d_view, rand_pool, 0.0, 1.0);

    // A = A_upper + A_upper^T + n*I
    Kokkos::parallel_for(
        "fill_A", n, KOKKOS_LAMBDA(const I i) {
          for (int j = 0; j < i + 1; j++) {
            d_A(j, i) = d_A(i, j);
            d_B(j, i) = d_B(i, j);
            if (i == j) {
              d_A(i, j) += n;
              d_B(i, j) += n;
            }
          }
        });

    A.modify_device();
    B.modify_device();

    A.sync_host();
    B.sync_host();

    printf("h_A(0, 0) = %f, h_B(0, 0) = %f\n", h_A(0, 0), h_B(0, 0));
    Kokkos::parallel_for(
        "print device", 1, KOKKOS_LAMBDA(const I i) {
          printf("d_A(0, 0) = %f, d_B(0, 0) = %f\n", d_A(i, i), d_B(i, i));
        });
  }

  Kokkos::DualView<T**> A;
  Kokkos::DualView<T**> B;
};

class FixSysPosMat : public ::testing::Test {
 public:
  //  protected:
  virtual void SetUp() override {
    A = Kokkos::DualView<T**>("A", 10, 10);
    B = Kokkos::DualView<T**>("B", 10, 10);
    I = Kokkos::DualView<T**>("I", 10, 10);
    Ainv = Kokkos::DualView<T**>("inverse(A)", 10, 10);
    w = Kokkos::DualView<T*>("eigenvalues", 10);
    v = Kokkos::DualView<T**>("eigenvectors", 10, 10);

    T A_data[10][10] = {{4.61107843, 0.55522023, -0.47604613, 0.66740675, -0.08236241, 0.77442043,
                         -0.16847107, 0.08554566, -0.1860363, -0.89421982},
                        {0.55522023, 7.57783175, -0.96999787, 0.41899955, -0.74169562, -0.83982537,
                         -0.16069949, 0.89837447, -0.12617254, 0.33375998},
                        {-0.47604613, -0.96999787, 6.25321263, -1.04613187, 0.48408668, -0.24834547,
                         -0.9308837, 0.55077296, 0.10224118, -0.79222648},
                        {0.66740675, 0.41899955, -1.04613187, 6.48985449, -0.21958628, 0.15202723,
                         1.15379689, 0.98174814, 0.75593, -0.81617607},
                        {-0.08236241, -0.74169562, 0.48408668, -0.21958628, 7.34816783, 0.46234805,
                         -0.13316185, -0.45256164, -0.23745895, -0.11623706},
                        {0.77442043, -0.83982537, -0.24834547, 0.15202723, 0.46234805, 6.69636965,
                         -0.58163821, 0.43759353, 0.05546983, -0.4466029},
                        {-0.16847107, -0.16069949, -0.9308837, 1.15379689, -0.13316185, -0.58163821,
                         5.23492206, 0.04704558, -0.47210199, 0.33053751},
                        {0.08554566, 0.89837447, 0.55077296, 0.98174814, -0.45256164, 0.43759353,
                         0.04704558, 6.63828556, -0.03473362, 0.05309124},
                        {-0.1860363, -0.12617254, 0.10224118, 0.75593, -0.23745895, 0.05546983,
                         -0.47210199, -0.03473362, 8.48108133, -1.02463494},
                        {-0.89421982, 0.33375998, -0.79222648, -0.81617607, -0.11623706, -0.4466029,
                         0.33053751, 0.05309124, -1.02463494, 7.15113258}};

    T B_data[10][10] = {
        {1.40247602e+00, 9.71640121e-02, -9.33275114e-02, -7.08290225e-02, 3.75605730e-04,
         -1.58986854e-01, 4.69279444e-02, -1.50059034e-01, 1.59782560e-01, 9.23926265e-02},
        {9.71640121e-02, 1.63109948e+00, 3.97920065e-02, 5.52071516e-02, -6.60095576e-02,
         -2.08674272e-02, 7.29017308e-02, -1.11165631e-03, 3.95364128e-02, -2.26175451e-01},
        {-9.33275114e-02, 3.97920065e-02, 1.51830773e+00, 9.84590157e-02, 7.29508523e-02,
         -5.51455823e-02, -2.53701148e-02, -8.70912341e-02, -4.21334552e-03, -5.38162686e-02},
        {-7.08290225e-02, 5.52071516e-02, 9.84590157e-02, 1.58193477e+00, -1.59398144e-01,
         -5.06519711e-03, -9.63670377e-03, -2.16515458e-01, -3.26136823e-02, -1.89053463e-02},
        {3.75605730e-04, -6.60095576e-02, 7.29508523e-02, -1.59398144e-01, 1.65732476e+00,
         -9.63299199e-02, 1.07806411e-01, 9.54309750e-02, 2.00620402e-02, -1.88117418e-02},
        {-1.58986854e-01, -2.08674272e-02, -5.51455823e-02, -5.06519711e-03, -9.63299199e-02,
         1.51002727e+00, 1.43406417e-02, -1.51418907e-01, -1.58634934e-01, 7.17075180e-03},
        {4.69279444e-02, 7.29017308e-02, -2.53701148e-02, -9.63670377e-03, 1.07806411e-01,
         1.43406417e-02, 1.33844572e+00, -1.01970936e-01, -1.91239978e-02, -2.87536341e-02},
        {-1.50059034e-01, -1.11165631e-03, -8.70912341e-02, -2.16515458e-01, 9.54309750e-02,
         -1.51418907e-01, -1.01970936e-01, 1.50181428e+00, -3.84895420e-02, 7.32017571e-02},
        {1.59782560e-01, 3.95364128e-02, -4.21334552e-03, -3.26136823e-02, 2.00620402e-02,
         -1.58634934e-01, -1.91239978e-02, -3.84895420e-02, 1.42023128e+00, 1.02036842e-01},
        {9.23926265e-02, -2.26175451e-01, -5.38162686e-02, -1.89053463e-02, -1.88117418e-02,
         7.17075180e-03, -2.87536341e-02, 7.32017571e-02, 1.02036842e-01, 1.47385147e+00}};

    T Ainv_data[10][10] = {
        {2.34242005e-01, -1.79757672e-02, 1.53928062e-02, -1.96061051e-02, 1.99662935e-03,
         -2.58916289e-02, 1.03306194e-02, 2.63010150e-03, 1.07542921e-02, 2.90568975e-02},
        {-1.79757672e-02, 1.42979200e-01, 2.32977581e-02, -3.63716368e-03, 1.00635187e-02,
         2.23441094e-02, 1.21898645e-02, -2.13339848e-02, 1.85839380e-03, -5.33484354e-03},
        {1.53928062e-02, 2.32977581e-02, 1.79668060e-01, 2.69974943e-02, -9.97494703e-03,
         1.33845734e-02, 2.71802251e-02, -2.41932425e-02, -1.16530272e-05, 2.34183538e-02},
        {-1.96061051e-02, -3.63716368e-03, 2.69974943e-02, 1.75958990e-01, 4.00394277e-04,
         -1.17299229e-03, -3.73433774e-02, -2.73961165e-02, -1.62087649e-02, 2.03318719e-02},
        {1.99662935e-03, 1.00635187e-02, -9.97494703e-03, 4.00394277e-04, 1.39155771e-01,
         -9.44359456e-03, 1.26102430e-03, 9.49729247e-03, 4.45409622e-03, 9.02137512e-04},
        {-2.58916289e-02, 2.23441094e-02, 1.33845734e-02, -1.17299229e-03, -9.44359456e-03,
         1.59402802e-01, 1.97272498e-02, -1.49664805e-02, 1.73782473e-04, 6.09413725e-03},
        {1.03306194e-02, 1.21898645e-02, 2.71802251e-02, -3.73433774e-02, 1.26102430e-03,
         1.97272498e-02, 2.08735668e-01, -1.07999526e-03, 1.40968326e-02, -6.89586142e-03},
        {2.63010150e-03, -2.13339848e-02, -2.41932425e-02, -2.73961165e-02, 9.49729247e-03,
         -1.49664805e-02, -1.07999526e-03, 1.61257597e-01, 2.71041272e-03, -6.02164784e-03},
        {1.07542921e-02, 1.85839380e-03, -1.16530272e-05, -1.62087649e-02, 4.45409622e-03,
         1.73782473e-04, 1.40968326e-02, 2.71041272e-03, 1.22515322e-01, 1.63727068e-02},
        {2.90568975e-02, -5.33484354e-03, 2.34183538e-02, 2.03318719e-02, 9.02137512e-04,
         6.09413725e-03, -6.89586142e-03, -6.02164784e-03, 1.63727068e-02, 1.51739942e-01}};

    T w_data[10] = {2.18001266, 2.51990827, 2.75329708, 3.90893235, 4.2813551,
                    4.9965097,  5.0351763,  5.94017474, 6.54900011, 8.35773};

    T v_data[10][10] = {{-0.36568376, 0.53458882, -0.08425405, 0.11737942, -0.26585255, -0.007811,
                         -0.00847441, 0.36821745, -0.19776121, 0.25947002},
                        {0.16509909, 0.06659396, 0.28786448, -0.17570303, -0.26259852, -0.49381132,
                         -0.06231013, -0.15836076, -0.3098011, -0.22694697},
                        {0.26315862, 0.35801608, 0.08802997, -0.27639771, 0.24018532, -0.12177436,
                         0.45021287, 0.23019121, 0.23058159, 0.0886961},
                        {0.36555391, 0.11324762, -0.3575999, -0.19256977, 0.04200757, 0.19413537,
                         -0.20088417, -0.10443037, -0.40399963, 0.30199261},
                        {-0.12723799, -0.01985568, 0.09358583, -0.43921288, 0.18505825, -0.06181628,
                         -0.55489877, 0.21373643, 0.09313537, 0.03035562},
                        {0.24401814, -0.0509544, 0.22930261, 0.43029066, 0.20311277, -0.26908234,
                         -0.19532623, 0.29543772, -0.05026158, 0.38583235},
                        {-0.04478387, 0.18512659, 0.56313051, 0.05254134, 0.25405063, 0.47731877,
                         0.06861497, -0.19096077, -0.26317057, -0.00881083},
                        {-0.29163712, -0.26587136, -0.02830397, -0.15118387, 0.25446925,
                         -0.14720062, 0.33181109, 0.08759725, -0.44577622, 0.35514387},
                        {-0.1255203, 0.13089124, 0.03983545, 0.03173073, 0.12574469, -0.21664864,
                         -0.07150045, -0.58656696, 0.26979744, 0.45266803},
                        {-0.03631802, 0.26115449, -0.21292441, 0.16526491, 0.4793798, -0.24723653,
                         -0.09602952, -0.09375213, -0.18601902, -0.46715739}};
    auto h_A = A.h_view;
    auto h_B = B.h_view;
    auto h_I = I.h_view;
    auto h_Ainv = Ainv.h_view;
    auto h_w = w.h_view;
    auto h_v = v.h_view;

    // copy data to Kokkos Views
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 10; j++) {
        h_A(i, j) = A_data[i][j];
        h_B(i, j) = B_data[i][j];
        h_Ainv(i, j) = Ainv_data[i][j];
        h_v(i, j) = v_data[i][j];
        if (i == j) {
          h_w(i) = w_data[i];
          h_I(i, j) = 1.0;
        }
      }
    }

    A.modify_host();
    B.modify_host();
    I.modify_host();
    Ainv.modify_host();
    w.modify_host();
    v.modify_host();

    A.sync_device();
    B.sync_device();
    I.sync_device();
    Ainv.sync_device();
    w.sync_device();
    v.sync_device();
  }

  Kokkos::DualView<T**> A;
  Kokkos::DualView<T**> B;
  Kokkos::DualView<T**> I;
  Kokkos::DualView<T**> Ainv;
  Kokkos::DualView<T*> w;
  Kokkos::DualView<T**> v;  // eigenvectors in column-major
};

// TEST_F(FixSysPosMat, sygvx) {
//   int M = 2;
//   int N = 10;
//   View1D<T> w_test("w_test", M);
//   View2D<T> v_test("v_test", N, M);

//   View2D<T> X("X", N, m);

//   linalg::lobpcg<T>(A.data(), B.data(), N, M, w_test.data(), v_test.data());

//   // printmat("w", w.data(), m, 1);
//   // printmat("w_test", w_test.data(), m, 1);

//   // printmat("v", v.data(), n, m, 1);
//   // printmat("v_test", v_test.data(), n, m);

//   // compare w to w_test
//   for (int i = 0; i < M; i++) {
//     EXPECT_NEAR(w(i), w_test(i), tol);
//   }

//   // compare the abs value for v to v_test (only first m columns)
//   for (int i = 0; i < N; i++) {
//     for (int j = 0; j < M; j++) {
//       EXPECT_NEAR(std::abs(v(i, j)), std::abs(v_test(i, j)), tol);
//     }
//   }
// }

// TEST_F(RandSysPosMat, lobpcg) {
//   View1D<T> w_test("w_test", m);
//   View2D<T> v_test("v_test", n, m);

//   View2D<T> X("X", n, m);

//   tick("linalg::lobpcg");
//   linalg::lobpcg<T>(A.data(), B.data(), n, m, w_test.data(), v_test.data());
//   tock("linalg::lobpcg");

//   // printmat("w", w_test.data(), m, 1);

//   // printmat("w", w.data(), m, 1);
//   // printmat("w_test", w_test.data(), m, 1);

//   // printmat("v", v.data(), n, m, 1);
//   // printmat("v_test", v_test.data(), n, m);
// }

// TEST_F(FixSysPosMat, lobpcg2) {
//   const int n = 10;
//   const int m = 1;
//   View1D<T> w_test("w_test", m);
//   View2D<T> v_test("v_test", n, m);

//   // View2D<T> X("X", n, m);

//   tick("linalg::lobpcg2");
//   linalg::lobpcg<T>(A.d_view.data(), B.d_view.data(), n, m, w_test.data(), v_test.data());
//   tock("linalg::lobpcg2");

//   // printmat("w", w.data(), m, 1);
//   // printmat("w_test", w_test.data(), m, 1);

//   // printmat("v", v.data(), n, m, 1);
//   // printmat("v_test", v_test.data(), n, m);
// }

TEST_F(RandSysPosMat, lobpcg2) {
  View1D<T> w_test("w_test", m);
  View2D<T> v_test("v_test", n, m);

  View2D<T> X("X", n, m);
  // View2D<T> Ainv("Ainv", n, n);
  // lapackage::inverse(B.h_view.data(), n, Ainv.data());

  // check();
  tick("linalg::lobpcg2");
  linalg::lobpcg<T>(A.d_view.data(), B.d_view.data(), n, m, w_test.data(), v_test.data());
  tock("linalg::lobpcg2");

  // View1D<T> w("w", m);
  // View2D<T> v("v", n, m);

  // tick("linalg::sygvx");
  // lapackage::sygvx<T>(A.data(), B.data(), n, m, w.data(), v.data());
  // tock("linalg::sygvx");

  // // compare w to w_test
  // for (int i = 0; i < m; i++) {
  //   EXPECT_NEAR(w_test(i), w(i), tol);
  // }

  // // compare the abs value for v to v_test, where v_test is in row-major and v
  // // is in column-major
  // for (int i = 0; i < n; i++) {
  //   for (int j = 0; j < m; j++) {
  //     EXPECT_NEAR(std::abs(v_test.data()[i * m + j]),
  //                 std::abs(v.data()[i + j * n]), tol);
  //   }
  // }

  // // printmat("w", w.data(), 1, m, std::make_pair(0, 0),
  // //          std::make_pair(0, m - 1));
  // // printmat("v", v.data(), n, m, std::make_pair(0, 5),
  // //          std::make_pair(m - 1, m - 1), 1);

  // printmat("w", w_test.data(), 1, m, std::make_pair(0, 0),
  //          std::make_pair(0, m - 1));
  // printmat("v", v_test.data(), n, m, std::make_pair(0, 5),
  //          std::make_pair(m - 1, m - 1));

  // View2D<T> AB("AB", 2, 4);
  // AB(0, 0) = 1;
  // AB(0, 1) = 1;
  // AB(0, 2) = 2;
  // AB(0, 3) = 2;
  // AB(1, 0) = 3;
  // AB(1, 1) = 3;
  // AB(1, 2) = 4;
  // AB(1, 3) = 4;
  // printmat("AB", AB.data(), 2, 4);

  // int n1 = 2;
  // int n2 = 3;
  // int stride1 = 1;
  // int stride2 = 2;
  // // Kokkos::LayoutStride layout(n1, 1, n2, 2);
  // Kokkos::LayoutStride layout(n1, 1, n2, 2);
  // Kokkos::View<T**, Kokkos::LayoutStride> a(AB.data(), layout);
  // // printmat("a", a.data(), 2, 4);
  // printf("a(0, 0) = %f\n", a(0, 0));
  // printf("a(0, 1) = %f\n", a(0, 1));
  // printf("a(0, 2) = %f\n", a(0, 2));
  // printf("a(0, 3) = %f\n", a(0, 3));
  // printf("a(1, 0) = %f\n", a(1, 0));
  // printf("a(1, 1) = %f\n", a(1, 1));
  // printf("a(1, 2) = %f\n", a(1, 2));
  // printf("a(1, 3) = %f\n", a(1, 3));

  // AB(0, 0) = 5;
  // printf("a(0, 0) = %f\n", a(0, 0));

  // printmat("a", a.data(), 8, 1);

  // auto a1 = Kokkos::subview(a, 1, Kokkos::ALL());
  // T sum = KokkosBlas::nrm2_squared(a1);
  // printf("sum = %f\n", sum);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);

  ::testing::InitGoogleTest(&argc, argv);

  // only run the selected tests
  // ::testing::GTEST_FLAG(filter) = "RandSysPosMat.lobpcg";
  // ::testing::GTEST_FLAG(filter) = "RandSysPosMat.lobpcg2";

  int result = RUN_ALL_TESTS();

  Kokkos::finalize();

  return result;
}