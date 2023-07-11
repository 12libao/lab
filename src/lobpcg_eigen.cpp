#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <iostream>
#include <random>

#include "utility.hpp"

using namespace Eigen;

Eigen::MatrixXd rand_symm_mat(int n = 10, double eig_low = 0.1,
                              double eig_high = 100.0, int nrepeat = 1) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distribution(eig_low, eig_high);

  Eigen::MatrixXd QRmat = Eigen::MatrixXd::Random(n, n);
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(QRmat);
  Eigen::MatrixXd Q = qr.householderQ();

  Eigen::VectorXd lam(n);
  if (nrepeat == 1) {
    lam = Eigen::VectorXd::NullaryExpr(n, [&]() { return distribution(gen); });
  } else {
    lam.head(nrepeat).fill(eig_low);
    lam.tail(n - nrepeat) = Eigen::VectorXd::NullaryExpr(
        n - nrepeat, [&]() { return distribution(gen); });
  }

  return Q * lam.asDiagonal() * Q.transpose();
}

std::pair<MatrixXd, MatrixXd> lobpcg(MatrixXd A, MatrixXd X,
                                     MatrixXd B = MatrixXd(),
                                     MatrixXd M = MatrixXd(), double tol = 1e-8,
                                     int maxiter = 200) {
  int N = X.rows();
  int m = X.cols();
  int maxiters = 200;

  // check();

  if (B.size() == 0) {
    B = MatrixXd::Identity(N, N);
  }

  MatrixXd P = MatrixXd::Zero(N, m);
  double residual = 1;
  int k = 0;
  VectorXd lambda_;
  MatrixXd Y;
  // check();

  VectorXd XBX = VectorXd::Zero(m);
  VectorXd XAX = VectorXd::Zero(m);
  VectorXd mu = VectorXd::Zero(m);

  // printf("Starting LOBPCG\n");
  // std::cout << "X = \n";
  // std::cout << X << "\n";
  // std::cout << "B = \n";
  // std::cout << B << "\n";

  while (k < maxiters && residual > tol) {
    // check();
    MatrixXd AX = A * X;
    MatrixXd BX = B * X;

    // std::cout << "AX = \n";
    // std::cout << AX << "\n";

    // std::cout << "BX = \n";
    // std::cout << BX << "\n";

    // check();
    // Compute the inner product for each column of X
    for (int i = 0; i < m; i++) {
      XBX(i) = (X.col(i).transpose() * BX.col(i))(0);
      XAX(i) = (X.col(i).transpose() * AX.col(i))(0);
      mu(i) = XBX(i) / XAX(i);
    }
    // std::cout << "mu = \n";
    // std::cout << mu << "\n";
    // check();
    // Compute residuals
    MatrixXd R = BX - AX * mu.asDiagonal();
    // std::cout << "R = \n";
    // std::cout << R << "\n";

    MatrixXd TR = M * R;
    // std::cout << "TR = \n";
    // std::cout << TR << "\n";

    MatrixXd W = TR;

    // std::cout << "W = \n";
    // std::cout << W << "\n";

    // std::cout << "X = \n";
    // std::cout << X << "\n";

    // std::cout << "P = \n";
    // std::cout << P << "\n";

    // Perform Rayleigh-Ritz procedure
    // Compute symmetric Gram matrices
    MatrixXd Z;
    if (k > 0) {
      Z = MatrixXd::Zero(N, 3 * m);
      Z << W, X, P;
    } else {
      Z = MatrixXd::Zero(N, 2 * m);
      Z << W, X;
    }

    // std::cout << "Z = \n";
    // std::cout << Z << "\n";

    // check();
    MatrixXd gramA = Z.transpose() * (A * Z);
    // check();
    MatrixXd gramB = Z.transpose() * (B * Z);
    // check();

    // std::cout << "gramA = \n";
    // std::cout << gramA << "\n";
    // std::cout << "gramB = \n";
    // std::cout << gramB << "\n";

    // Solve generalized eigenvalue problem
    GeneralizedSelfAdjointEigenSolver<MatrixXd> eigensolver(gramA, gramB);
    // check();
    // Y = first m eigenvectors
    Y = eigensolver.eigenvectors().leftCols(m);
    lambda_ = eigensolver.eigenvalues().head(m);
    // check();

    // std::cout << "Y = \n";
    // std::cout << Y << "\n";

    // Compute Ritz vectors
    MatrixXd Yw = Y.topRows(m);
    MatrixXd Yx = Y.middleRows(m, m);
    MatrixXd Yp;
    if (k > 0) {
      Yp = Y.bottomRows(m);
    } else {
      Yp = MatrixXd::Zero(m, m);
    }

    // std::cout << "Yw = \n";
    // std::cout << Yw << "\n";
    // std::cout << "Yx = \n";
    // std::cout << Yx << "\n";
    // std::cout << "Yp = \n";
    // std::cout << Yp << "\n";

    // check();
    X = W * Yw + X * Yx + P * Yp;
    P = W * Yw + P * Yp;

    // std::cout << "X = \n";
    // std::cout << X << "\n";

    // std::cout << "P = \n";
    // std::cout << P << "\n";

    residual = R.norm() / A.norm();
    k++;

    printf("Iteration %d, residual = %e\n", k, residual);
  }
  // check();

  MatrixXd eigenval = lambda_;
  MatrixXd eigenvec = X;
  return {eigenval, eigenvec};
}

int main() {
  int n = 10000;
  int m = 5;
  check();
  MatrixXd A = rand_symm_mat(n, 2.0, 10.0);
  check();
  MatrixXd B = rand_symm_mat(n, 1.0, 2.0);
  check();
  // MatrixXd A = MatrixXd::Random(n, n);
  // A = A.transpose() * A;  // make positive efinite
  // A = A + A.transpose() + MatrixXd::Identity(n, n) * 0.1;  // make symmetric
  // check();
  // MatrixXd B = MatrixXd::Random(n, n);
  // B = B.transpose() * B;  // make positive definite
  // B = B + B.transpose();  // make symmetric
  MatrixXd X = MatrixXd::Random(n, m);
  check();
  MatrixXd M = A.inverse();

  // // find the eigenvalues for A
  // EigenSolver<MatrixXd> es(A);
  // std::cout << "The eigenvalues of A are:\n" << es.eigenvalues().real() <<
  // "\n";
  check();
  tick("lobpcg");
  auto result = lobpcg(A, X, B, M);
  tock("lobpcg");
  check();
  std::cout << "Eigenvalues:\n" << result.first << "\n";
  std::cout << "Eigenvectors:\n"
            << result.second.leftCols(1).topRows(5) << "\n";

  return 0;
}