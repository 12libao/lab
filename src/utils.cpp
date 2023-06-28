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

#endif  // TOOLS_H