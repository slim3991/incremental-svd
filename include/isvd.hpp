#pragma once
#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/QR"
#include "Eigen/SVD"

template <typename scalar> class IncrementalSVD {
  typedef Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> EigMatrix;
  typedef Eigen::Vector<scalar, Eigen::Dynamic> EigVector;

public:
  EigMatrix U;
  EigVector S;

private:
  bool is_fitted = false;
  int rank;
  scalar f; // forgetting factor
  int time_since_reorth = 0;

public:
  IncrementalSVD(int r, scalar ff = 1.0) : rank{r}, f{ff} {}

  void fit(EigMatrix X) {
    Eigen::BDCSVD<EigMatrix> svd(X, Eigen::ComputeThinU);
    U = svd.matrixU().leftCols(rank);
    S = svd.singularValues().head(rank);
    is_fitted = true;
  }

  void increment(EigVector new_vec) {
    EigVector m = U.transpose() * new_vec;
    EigVector p = new_vec - (U * m);
    scalar p_norm = p.norm();

    if (p_norm > 1e-10) {
      EigVector q = p / p_norm;

      EigMatrix K = EigMatrix::Zero(S.size() + 1, S.size() + 1);
      K.block(0, 0, S.size(), S.size()) = (S * f).asDiagonal();
      K.block(0, S.size(), S.size(), 1) = m;
      K(S.size(), S.size()) = p_norm;

      Eigen::JacobiSVD<EigMatrix> svd(K, Eigen::ComputeFullU);

      EigMatrix combined_U = EigMatrix(U.rows(), U.cols() + 1);
      combined_U << U, q;
      U = (combined_U * svd.matrixU().leftCols(rank));
      S = svd.singularValues().head(rank);

      // re-orthogonalize
      Eigen::HouseholderQR<EigMatrix> qr(U);
      U = qr.householderQ() * EigMatrix::Identity(U.rows(), U.cols());
    }
  }

  // Bathed incrementer
  void increment(EigMatrix new_mat) {
    for (int i = 0; i < new_mat.cols(); ++i) {
      // copy was necessary here due to Eigen wierdness, bottlenneck is 
      // svd so this is fine.
      EigVector col_vec = new_mat.col(i);
      this->increment(col_vec);
    }
  }
};
