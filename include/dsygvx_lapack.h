#ifndef DSYGVX_LAPACK_H
#define DSYGVX_LAPACK_H

#include <Accelerate/Accelerate.h>

#include <vector>

extern "C" {
int dsygvx_(int* itype, char* jobz, char* range, char* uplo, int* n, double* a,
            int* lda, double* b, int* ldb, double* vl, double* vu, int* il,
            int* iu, double* abstol, int* m, double* w, double* z, int* ldz,
            double* work, int* lwork, int* iwork, int* ifail, int* info);
}

std::pair<std::vector<double>, std::vector<double>> dsygvx(
    std::vector<double>& A, std::vector<double>& B, int n, int m);

#endif  // DSYGVX_LAPACK_H