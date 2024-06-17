#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <eigen3/Eigen/Dense>

#include "NNAD/FeedForwardNN.h"
#include "NTK/utility_test.hpp"
#include "NTK/NumericalDerivative.h"
#include "NTK/utility_test.hpp"

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


TEST_CASE( " Test first order num. derivative ", "[Derivatives][First] ") {

  SECTION(" Test against polynomial ", "[Derivatives][First][Polynomial]") {

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

  SECTION( " Test with neural network " ) {

    const std::vector<int> arch{1, 20, 35, 1};
    std::unique_ptr<NTK::NNAD> nn = std::make_unique<NTK::NNAD>(arch, 0, false,
                                        nnad::Tanh<double>,  nnad::dTanh<double>,
                                        nnad::OutputFunction::LINEAR);
    KnownNNAD dummyNN(nn.get());
    NTK::data X {0.5};
    std::vector<double> NNADJacobian = NTK::dNNAD_cleaner(nn.get(), X);
    std::vector<double> nNNADJacobian = NTK::helper::HelperFirstFiniteDer(nn.get(), X, 1.e-5);

    REQUIRE( NNADJacobian.size() == nNNADJacobian.size() );
    for(size_t ip=0; ip<NNADJacobian.size(); ip++) {
      REQUIRE_THAT( NNADJacobian[ip], Catch::Matchers::WithinRel(nNNADJacobian[ip], 1.e-5) );
    }
  }
}

TEST_CASE( " Test second order num. derivative ", "[Derivatives][Second] " ) {

    SECTION(" Test against polynomial ", "[Derivatives][Seceond][Polynomial]") {

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

  SECTION( " Test with neural network " ) {

    const std::vector<int> arch{1, 10, 10, 1};
    std::unique_ptr<NTK::NNAD> nn = std::make_unique<NTK::NNAD>(arch, 0, false,
                                        nnad::Tanh<double>,  nnad::dTanh<double>,
                                        nnad::OutputFunction::LINEAR);
    NTK::data X {0.5};
    double eps1 = 1.e-4;

    SECTION(" Test ndd = nd(d) ") {
      std::vector<double> ndd_NNAD = NTK::helper::nddNNAD (nn.get(), X, eps1);
      std::vector<double> nd_dNNAD = NTK::helper::HelperSecondFiniteDer2 (nn.get(), X, eps1);

      REQUIRE( ndd_NNAD.size() == nd_dNNAD.size() );
      for(size_t ip=0; ip<nd_dNNAD.size(); ip++) {
        REQUIRE_THAT( ndd_NNAD[ip], Catch::Matchers::WithinAbs(nd_dNNAD[ip], 1.e-5) );
      }
    }
  }
}

TEST_CASE(" Test dummy neural network ") {

  const std::vector<int> arch{1, 2, 1};
  std::unique_ptr<NTK::NNAD> nn = std::make_unique<NTK::NNAD>(arch, 0, false,
                                        nnad::Tanh<double>,  nnad::dTanh<double>,
                                        nnad::OutputFunction::LINEAR);
  KnownNNAD dummyNN(nn.get());
  NTK::data X {0.5};
  const int np = nn->GetParameterNumber();

  SECTION(" Testing output ") {
    double dummyNNResult = dummyNN.PHI(X[0]);
    std::vector<double> NNADResult = nn->Evaluate(X);

    // Check both evaluate to the same value
    REQUIRE_THAT(dummyNNResult, Catch::Matchers::WithinRel(NNADResult[0]));
  }

  SECTION(" Testing Jacobian ") {
    // Check both evaluate to the same derivative
    std::vector<double> NNADJacobian = nn->Derive(X);
    NNADJacobian.erase(NNADJacobian.begin(), NNADJacobian.begin() + nn->GetArchitecture().back());
    std::vector<double> DummyJacobian = dummyNN.jacobian(X[0]);
    REQUIRE( NNADJacobian.size() == DummyJacobian.size() );
    for (size_t ip=0; ip < DummyJacobian.size(); ip++) {
      REQUIRE_THAT(DummyJacobian[ip], Catch::Matchers::WithinRel(NNADJacobian[KnownNNAD::DummyToNNAD.at(ip)]));
    }
  }

  SECTION(" Testing Hessian ") {
    std::vector<double> NNADHessian = NTK::helper::HelperSecondFiniteDer2(nn.get(), X, 1.e-6);
    Eigen::MatrixXd DummyHessian = dummyNN.Hessian(X[0]);

    for(size_t ip=0; ip<np; ip++) {
      for(size_t jp=0; jp<np; jp++) {
        int nip = KnownNNAD::DummyToNNAD.at(ip);
        int njp = KnownNNAD::DummyToNNAD.at(jp);
        double NNADres = NNADHessian[ njp + np * nip ];
        double DummyRes = DummyHessian(ip,jp);
        REQUIRE_THAT(DummyRes, Catch::Matchers::WithinRel(NNADres, eps_catch2));
      }
    }
  }

  SECTION(" Testing third derivative ") {
    std::vector<double> NNADThird = NTK::helper::HelperThirdFiniteDer(nn.get(), X, 1.e-5);
    Eigen::Tensor<double,3> DummyThird = dummyNN.ThirdDerivative(X[0]);

    for(size_t ip=0; ip<np; ip++) {
      for(size_t jp=0; jp<np; jp++) {
        for(size_t kp=0; kp<np; kp++) {
          int nip = KnownNNAD::DummyToNNAD.at(ip);
          int njp = KnownNNAD::DummyToNNAD.at(jp);
          int nkp = KnownNNAD::DummyToNNAD.at(kp);
          double NNAD_third = NNADThird[ nkp + np * njp + np * np * nip]; // row-major
          double DummyRes_third = DummyThird(ip,jp, kp);
          REQUIRE_THAT(DummyRes_third, Catch::Matchers::WithinAbs(NNAD_third, eps_catch2));
        }
      }
    }
  }

  SECTION(" Test third derivatives ") {
    INFO("To be implemented");
  }


  
}



