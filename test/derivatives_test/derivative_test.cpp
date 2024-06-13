#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <eigen3/Eigen/Dense>

#include "NNAD/FeedForwardNN.h"
#include "NTK/utility_test.hpp"
#include "NTK/NumericalDerivative.h"


class KnwonNNAD {
  public:
  KnwonNNAD(nnad::FeedForwardNN<double> *NN) {
    b1 = NN->GetBias(1,0);
    b2 = NN->GetBias(1,1);
    B  = NN->GetBias(2,0);
    w1 = NN->GetLink(1, 0, 0);
    w2 = NN->GetLink(1, 1, 0);
    W1 = NN->GetLink(2, 0, 0);
    W2 = NN->GetLink(2, 0, 1);
  }

  double tanh(double x) { return nnad::Tanh<double> (x); }
  double dtanh(double x) { return nnad::dTanh<double> (x); }
  double ddtanh(double x) { return -2. * tanh(x) * dtanh(x); }
  double dddtanh(double x) { return -2. * dtanh(x) * dtanh(x) - 2. * tanh(x) * ddtanh(x); }
  double phi_1 (double x) { return b1 + w1 * x; }
  double phi_2 (double x) { return b2 + w2 * x; }
  double PHI (double x) { return B + W1 * tanh(phi_1 (x)) + W2 * tanh(phi_2 (x)); }


  double H13 (double x) { return dtanh(phi_1(x)); }
  double H15 (double x) { return dtanh(phi_1(x)) * x; }
  double H24 (double x) { return dtanh(phi_2(x)); }
  double H26 (double x) { return dtanh(phi_2(x)) * x; }
  double H33 (double x) { return W1 * ddtanh(phi_1(x)); }
  double H35 (double x) { return W1 * x * ddtanh(phi_1(x)); }
  double H44 (double x) { return W2 * ddtanh(phi_2(x)); }
  double H46 (double x) { return W2 * x * ddtanh(phi_2(x)); }
  double H55 (double x) { return W1 * ddtanh(phi_1(x)) * x * x; }
  double H66 (double x) { return W2 * ddtanh(phi_2(x)) * x * x; }


  std::vector<double> jacobian(double x) {
    std::vector<double> jaco {1.0,
                              tanh(phi_1(x)),
                              tanh(phi_2(x)),
                              W1 * dtanh(phi_1(x)),
                              W2 * dtanh(phi_2(x)),
                              W1 * x * dtanh(phi_1(x)),
                              W2 * x * dtanh(phi_2(x))};
    return jaco;
  }

  Eigen::MatrixXd Hessian (double x) {

    Eigen::MatrixXd hes {
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
      };

      hes(1,3) = H13(x);
      hes(3,1) = hes(1,3);
      hes(1,5) = H15(x);
      hes(5,1) = hes(1,5);
      hes(2,4) = H24(x);
      hes(4,2) = hes(2,4);
      hes(2,6) = H26(x);
      hes(6, 2) = hes(2, 6);
      hes(3,3) = H33(x);
      hes(3,5) = H35(x);
      hes(5,3) = hes(3,5);
      hes(4,4) = H44(x);
      hes(4,6) = H46(x);
      hes(6,4) = hes(4,6);
      hes(5,5) = H55(x);
      hes(6,6) = H66(x);

    return hes;
  }

  double NTK (double x1, double x2) {
    std::vector<double> jaco1 = jacobian(x1);
    std::vector<double> jaco2 = jacobian(x2);
    double res = std::inner_product(jaco1.begin(), jaco1.end(), jaco2.begin(), 0.);
    return res;
  }

    double O3 (double x1, double x2, double x3) {
    std::vector<double> jaco1 = jacobian(x1);
    std::vector<double> jaco2 = jacobian(x2);
    double res = std::inner_product(jaco1.begin(), jaco1.end(), jaco2.begin(), 0.);
    return res;
  }

  private:
  double b1, b2, w1, w2, B, W1, W2;
};

std::function<std::vector<double> (std::vector<double> const&, std::vector<double>)> PolTestFunction
    {
      [] (std::vector<double> const& X, std::vector<double> parameters) -> std::vector<double>
      {
        double theta_1 = parameters[0];
        double theta_2 = parameters[1];
        double theta_3 = parameters[2];
        double x = X[0];
        std::vector<double> res{1. + theta_1 * x + pow(theta_2, 2) * pow(x, 3) + pow(theta_3, 4) * pow(x, 5)};
        return res;
      }
    };

double eps_catch2 = 1.e-5;

TEST_CASE( " Test polynomial " ) {
  
   SECTION( " Test polynomial integers ") {
      std::vector<double> pars {1., 1., 1.};
      std::vector<double> x {1.};
      std::vector<double> result = PolTestFunction(x, pars);
      REQUIRE_THAT(result[0], Catch::Matchers::WithinRel(4.));
    }

    SECTION( " Test polynomial double ") {
      std::vector<double> pars {1.7, 2.5, 3.7};
      std::vector<double> x {2.7};
      std::vector<double> result = PolTestFunction(x, pars);
      REQUIRE_THAT(result[0], Catch::Matchers::WithinRel(27020.77064, eps_catch2));
    }
}

TEST_CASE(" Test known neural network ") {

}


TEST_CASE( " Test first order num. derivative ", "[Derivatives][First] ") {

  SECTION( " Test against polynomial ", "[Derivatives][First][Polynomial]") {

    std::vector<double> pars {1.7, 2.5, 3.7};
    std::vector<double> X {1.5};
    double x = X[0];
    std::vector<double> jaco = NTK::FiniteDifferenceVec (PolTestFunction, pars, X, 1, 1.e-5);
    std::vector<double> truth ({x, 2. * pars[1] * pow(x, 3), 4. * pow(pars[2], 3) * pow(x, 5)});
    REQUIRE( jaco.size() == truth.size());
    for (size_t i = 0; i< jaco.size(); i++) {
      REQUIRE_THAT(jaco[i], Catch::Matchers::WithinRel(truth[i], eps_catch2));
    }
  }

  SECTION( " Test against known network " ) {

  }
}

TEST_CASE( " Test second order num. derivative ", "[Derivatives][Second] " ) {


    SECTION( " Second derivative ") {
      std::vector<double> pars {1.7, 2.5, 3.7};
      std::vector<double> X {1.5};
      double x = X[0];
      // This is ordered in col-major
      std::vector<double> hessian = NTK::FiniteSecondDifferenceVec (PolTestFunction, pars, X, 1, 1.e-4);

      // This in row major
      std::vector<double> truth {0., 0., 0.,
                                 0, 2. * pow(x,3), 0.,
                                 0., 0., 12. * pow(pars[2], 2) * pow(x, 5)};
      REQUIRE( hessian.size() == truth.size());
      for (size_t i = 0; i< pars.size(); i++) { // row
        for (size_t j = 0; j< pars.size(); j++) { // col
          INFO("Row " << i );
          INFO("Column " << j );
          REQUIRE_THAT(hessian[i + pars.size() * j], Catch::Matchers::WithinRel(truth[ j + pars.size() * i], eps_catch2));
        }
      }
  }
}
