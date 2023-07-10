#ifndef TOOLS_H
#define TOOLS_H

// Macro to print the line number with pass status
#define check() printLineStatus(__LINE__, true)
#define tick(msg) startTimer(msg);
#define tock(...) reportTimer(__VA_ARGS__);

// #include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::time_point<Clock> TimePoint;
typedef std::chrono::duration<double> Duration;

TimePoint startTime;
TimePoint endTime;

void startTimer(const char* msg) { startTime = Clock::now(); }

double getElapsedTime() {
  Duration d = Clock::now() - startTime;
  return d.count();
}

void reportTimer(const char* msg = "") {
  double elapsed = getElapsedTime();
  printf("%s: ", msg);
  // if (elapsed < 1e-6)
  //   printf("\033[32m%.5f us\033[0m\n", elapsed * 1e6);
  // else if (elapsed < 1e-3)
  //   printf("\033[32m%.5f ms\033[0m\n", elapsed * 1e3);
  // else
  printf("\033[32m%.8f s\033[0m\n", elapsed);
}

// Function to print the line number and status
void printLineStatus(int lineNumber, bool passed) {
  printf("Line: \033[32m%d ", lineNumber);
  if (passed) {
    printf("pass");
  } else {
    printf("not pass");
  }
  printf("\033[0m\n");
}

// /*
//  * Fill a nxn matrix with random values between min and max
//  *
//  * Input:
//  *   T A: nxn matrix       - row major
//  *   T min: minimum value  - default 0.0
//  *   T max: maximum value  - default 1.0
//  */
template <typename T, int N>
void randFill(std::array<std::array<T, N>, N>& A, T min = -1.0, T max = 1.0) {
  static std::random_device rd;   // only need to initialize it once
  static std::mt19937 mte(rd());  // this is a relative big object to create

  std::uniform_real_distribution<T> dist(min, max);

  for (int i = 0; i < N; ++i) {
    std::generate(A[i].begin(), A[i].end(), [&]() { return dist(mte); });
  }
}

template <typename container>
void printMat(const char* name, container& A, int N=5) {
  printf("Matrix: \033[32m%s\033[0m\n", name);
  for (int i = 0; i < N; ++i) {
    printf("  |");
    for (int j = 0; j < N; ++j) {
        printf("%9.5f ", A(i, j));
    }
    printf("|\n");
  }
  printf("\n");
}

// template <typename Container>
// void printMat(const char* name, Container& A, int N=5) {
//   printf("Matrix: \033[32m%s\033[0m\n", name);
//   for (auto it_row = A.begin(); it_row != A.end(); ++it_row) {
//     printf("  |");
//     for (auto it_col = it_row->begin(); it_col != it_row->end(); ++it_col) {
//         printf("%9.5f ", *it_col);
//     }
//     printf("|\n");
//   }
//   printf("\n");
// }


#endif  // TOOLS_H