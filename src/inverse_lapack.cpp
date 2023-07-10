#include <Accelerate/Accelerate.h>

#include <Kokkos_Core.hpp>  // include kokkos
#include <array>
#include <cstdio>

extern "C" {
// LU decomoposition of a general matrix
int dgetrf_(int* M, int* N, double* A, int* lda, int* IPIV, int* INFO);

// generate inverse of a matrix given its LU decomposition
int dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork,
            int* INFO);
}

void inverse(double* A, int N) {
  int* IPIV = new int[N];
  int LWORK = N * N;
  double* WORK = new double[LWORK];
  int INFO;

  dgetrf_(&N, &N, A, &N, IPIV, &INFO);
  dgetri_(&N, A, &N, IPIV, WORK, &LWORK, &INFO);

  delete[] IPIV;
  delete[] WORK;
}

int main() {
  Kokkos::initialize();
  {
    Kokkos::View<double**, Kokkos::HostSpace> A("A", 2, 2);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(1, 0) = 3;
    A(1, 1) = 4;

    // convert to C array
    double* A_ptr = A.data();

    inverse(A_ptr, 2);
    printf("%f %f\n", A(0, 0), A(0, 1));
    printf("%f %f\n", A(1, 0), A(1, 1));
  }
  Kokkos::finalize();
}