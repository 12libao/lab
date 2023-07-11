#ifndef LAPACKAGE_HPP
#define LAPACKAGE_HPP

#include <lapacke.h>

#include <array>
#include <cstdio>

// intialize a struct to hold the eigenvalues and eigenvectors
template <typename T, std::size_t N, std::size_t M>
struct EigenPair {
  std::array<T, M> eigenvalues;
  std::array<T, N * M> eigenvectors;
};

/*
 DSYGVX computes selected eigenvalues, and optionally, eigenvectors
 of a real generalized symmetric-definite eigenproblem, of the form
 A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.  Here A
 and B are assumed to be symmetric and B is also positive definite.
 Eigenvalues and eigenvectors can be selected by specifying either a
 range of values or a range of indices for the desired eigenvalues.
*/
template <typename T, std::size_t N, std::size_t M>
EigenPair<T, N, M> dsygvx(const std::array<T, N * N>& A,
                          const std::array<T, N * N>& B);

/*
  uses the LAPACK routine dgetrf and dgetri to compute the inverse of a matrix
*/
void inverse(double* A, int N);

#endif  // LAPACKAGE_HPP