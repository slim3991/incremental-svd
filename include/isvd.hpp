#include "../extern/Eigen/Dense"
#include "../extern/Eigen/QR"
#include "../extern/Eigen/SVD"

class IncrementalSVD {
public:
  Eigen::MatrixXd U;
  Eigen::VectorXd S;

private:
  bool is_fitted = false;
  int rank;
  float f; // forgetting factor
  int time_since_reorth = 0;

public:
  IncrementalSVD(int r, float ff = 1.0) : rank{r}, f{ff} {}

  void fit(Eigen::MatrixXd X) {
    Eigen::BDCSVD<Eigen::MatrixXd> svd(X, Eigen::ComputeThinU);
    int current_rank =
        std::min(rank, static_cast<int>(svd.singularValues().size()));

    U = svd.matrixU();
    S = svd.singularValues().head(current_rank);
    is_fitted = true;
  }

  void increment(Eigen::VectorXd new_vec) {
    Eigen::VectorXd m = U.transpose() * new_vec;
    Eigen::VectorXd p = new_vec - (U * m);
    double p_norm = p.norm();

    if (p_norm > 1e-10) {
      Eigen::VectorXd q = p / p_norm;

      Eigen::MatrixXd K = Eigen::MatrixXd::Zero(S.size() + 1, S.size() + 1);
      K.block(0, 0, S.size(), S.size()) = (S * f).asDiagonal();
      K.block(0, S.size(), S.size(), 1) = m;
      K(S.size(), S.size()) = p_norm;

      Eigen::JacobiSVD<Eigen::MatrixXd> svd(K, Eigen::ComputeFullU);

      Eigen::MatrixXd combined_U = Eigen::MatrixXd(U.rows(), U.cols() + 1);
      combined_U << U, q;
      U = (combined_U * svd.matrixU().leftCols(rank));
      S = svd.singularValues().head(rank);
    }

    if (++time_since_reorth >= 10) {
      re_orth();
      time_since_reorth = 0;
    }
  }

private:
  void re_orth() {
    Eigen::FullPivHouseholderQR<Eigen::MatrixXd> qr(U);
    U = qr.matrixQ();
  }
};
