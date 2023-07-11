#include <iostream> // std::cout
#include <vector>   // std::vector
#include <algorithm> // std::sort
#include <iterator> // std::ostream_iterator
#include <array> // std::array
// include kokkos
#include <Kokkos_Core.hpp>
// include kokkoskernels
#include <KokkosBlas1_axpby.hpp>
// include MPI
#include <mpi.h>
// include openmp
#include <omp.h>
// include lobpcg
#include "lobpcg.hpp"

int main(int argc, char* argv[]) {
  // initialize kokkos
  Kokkos::initialize(argc, argv);
  {
    // create a vector of 10 elements
    std::vector<double> x(10, 1.0);
    // create a vector of 10 elements
    std::vector<double> y(10, 2.0);
    // create a vector of 10 elements
    std::vector<double> z(10, 0.0);
    // create a kokkos view of x
    Kokkos::View<double*> x_view(x.data(), x.size());
    // create a kokkos view of y
    Kokkos::View<double*> y_view(y.data(), y.size());
    // create a kokkos view of z
    Kokkos::View<double*> z_view(z.data(), z.size());
    // compute z = 2.0 * x + y
    KokkosBlas::axpby(2.0, x_view, 1.0, y_view);
    // print z
    std::cout << "z = ";
    std::copy(z.begin(), z.end(), std::ostream_iterator<double>(std::cout, " "));
    std::cout << std::endl;
  }
  // finalize kokkos
  Kokkos::finalize();
  return 0;
}

