#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include "NTK/Observable.h"
#include "NNAD/FeedForwardNN.h"
#include "NTK/utility_test.hpp"

//                            Basic Observable
//_______________________________________________________________________________________
TEMPLATE_TEST_CASE( " Testing basic observable", "[Observable][Basic][template] ", NTK::dNN, NTK::ddNN ) {
  INFO("Initialising neural network and data");
  const std::vector<int> arch{3, 5, 2};
  std::unique_ptr<NTK::NNAD> nn = std::make_unique<NTK::NNAD>(arch, 0, nnad::OutputFunction::QUADRATIC, false);
  int nout = arch.back();
  int np = nn->GetParameterNumber();
  int batch_size = 4;
  std::vector<NTK::data> data_vector (batch_size);
  data_vector[0] = {0.0, 0.3, 0.6};
  data_vector[1] = {0.0, 0.7, 0.9};
  data_vector[2] = {0.8, 0.1, 0.3};
  data_vector[3] = {0.1, 0.9, 0.5};

  SECTION( "Testing constructor" ) {
    TestType dnn (batch_size, nout, np);
    auto tensor = dnn.GetTensor();
    Eigen::Tensor<double, 0> result = tensor.sum();
    REQUIRE_THAT(result(0), Catch::Matchers::WithinRel(0.));
  }

  SECTION( " Testing evaluation , `data_map`, and`is_computed` " ) {
    std::unique_ptr<TestType> dnn = std::make_unique<TestType> (batch_size, nout, np);
    int i = 3;

    SECTION( " Testing Evaluation ") {
      REQUIRE_NOTHROW(dnn->Evaluate(data_vector[i], i , nn.get()));
      REQUIRE_NOTHROW(dnn->Evaluate(data_vector, nn.get()));
    }

    SECTION( " Testing data_map ") {
      INFO("Checking if data_map is empty before evaluation");
      REQUIRE(dnn->GetDataMap().empty());

      INFO("Checking if data_map is empty after single evaluation");
      dnn->Evaluate(data_vector[i], i , nn.get());
      REQUIRE_FALSE(dnn->GetDataMap().empty());

      INFO("Check map has been indexed correcly");
      REQUIRE(dnn->GetDataMap().count(i) == 1);
      REQUIRE(dnn->GetDataMap().count(0) == 0);
      REQUIRE(dnn->GetDataMap().count(1) == 0);
      REQUIRE(dnn->GetDataMap().count(2) == 0);
      REQUIRE(dnn->GetDataMap().size() == 1);
    }

    SECTION( " Testing `is_copmuted` ") {
      REQUIRE_FALSE(dnn->is_computed());
      dnn->Evaluate(data_vector , nn.get());
      REQUIRE(dnn->is_computed());
    }

    SECTION( "Testing against truth" ) {
      double truth;
      if (TestType::id == "NTK::dNN")
        truth = sum_utility_return_dNN(batch_size, nout, np, nn.get(), data_vector);
      else if (TestType::id == "NTK::ddNN")
        truth = sum_utility_return_ddNN(batch_size, nout, np, nn.get(), data_vector);

      SECTION( " Iterative " )   {
        for (int a = 0; a < data_vector.size(); a++) {
          dnn->Evaluate(data_vector[a], a, nn.get());
        }
        Eigen::Tensor<double, 0> guess = dnn->GetTensor().sum();
        REQUIRE_THAT(guess(0), Catch::Matchers::WithinRel(truth));
      }

      SECTION( " Vector " )   {
        dnn->Evaluate(data_vector, nn.get());
        Eigen::Tensor<double, 0> guess = dnn->GetTensor().sum();
        REQUIRE_THAT(guess(0), Catch::Matchers::WithinRel(truth));
      }

    } 

    SUCCEED( " Evaluation test succeeded" );
  }
  SUCCEED( " Observable test succeeded" );
}



//                            Combined Observable
//_______________________________________________________________________________________
TEST_CASE( " Testing combined observable", "[Observable][Combined][template] ") {
  const std::vector<int> arch{3, 5, 2};
  std::unique_ptr<NTK::NNAD> nn = std::make_unique<NTK::NNAD>(arch, 0, nnad::OutputFunction::QUADRATIC, false);
  int nout = arch.back();
  int np = nn->GetParameterNumber();
  int batch_size = 4;
  std::vector<NTK::data> data_vector (batch_size);
  data_vector[0] = {0.0, 0.3, 0.6};
  data_vector[1] = {0.0, 0.7, 0.9};
  data_vector[2] = {0.8, 0.1, 0.3};
  data_vector[3] = {0.1, 0.9, 0.5};

  SECTION( "Testing constructors" ) {

    SECTION( " Testing constructors O2 ") {
      NTK::O2 ntk (batch_size, nout);
      auto tensor = ntk.GetTensor();
      Eigen::Tensor<double, 0> result = tensor.sum();
      REQUIRE_THAT(result(0), Catch::Matchers::WithinRel(0.));
    }

    SECTION( " Testing constructors O3 ") {
      NTK::O3 O (batch_size, nout);
      auto tensor = O.GetTensor();
      Eigen::Tensor<double, 0> result = tensor.sum();
      REQUIRE_THAT(result(0), Catch::Matchers::WithinRel(0.));
    }
  }

  SECTION( " Testing contraction and `check_observables` " ) {

    SECTION(" Test for O2") {
      NTK::O2 O (batch_size, nout);
      NTK::dNN dnn(batch_size, nout, np);
      REQUIRE_THROWS(O.Evaluate(&dnn));
      dnn.Evaluate(data_vector, nn.get());
      REQUIRE_NOTHROW(O.Evaluate(&dnn));
    }

    SECTION( " Test for O3 ") {
      NTK::O3 O (batch_size, nout);
      NTK::dNN dnn (batch_size, nout, np);
      NTK::ddNN ddnn (batch_size, nout, np);
      REQUIRE_THROWS(O.Evaluate(&dnn, &ddnn));
      dnn.Evaluate(data_vector, nn.get());
      ddnn.Evaluate(data_vector, nn.get());
      REQUIRE_NOTHROW(O.Evaluate(&dnn, &ddnn));
    }
  }

  SECTION( " Testing against old implementation ") {

    SECTION( " Test for O2 ") {
      NTK::O2 ntk (batch_size, nout);
      NTK::dNN dnn(batch_size, nout, np);
      dnn.Evaluate(data_vector, nn.get());
      ntk.Evaluate(&dnn);
      auto dnn_tensor = dnn.GetTensor();
      auto ntk_tensor = ntk.GetTensor();

      // Contraction old way
      Eigen::array<Eigen::IndexPair<int>, 1> double_contraction = { Eigen::IndexPair<int>(2,2) };
      NTK::Tensor<4> truth_tensor = dnn_tensor.contract(dnn_tensor, double_contraction);

      Eigen::Tensor<double, 0> guess = ntk_tensor.sum();
      Eigen::Tensor<double, 0> target = truth_tensor.sum();
      REQUIRE_THAT(guess(0), Catch::Matchers::WithinRel(target(0)));
    }

    SECTION( " Test for O3 ") {
      NTK::O3 ntk_3 (batch_size, nout);
      NTK::dNN dnn(batch_size, nout, np);
      NTK::ddNN ddnn(batch_size, nout, np);
      dnn.Evaluate(data_vector, nn.get());
      ddnn.Evaluate(data_vector, nn.get());
      ntk_3.Evaluate(&dnn, &ddnn);
      auto dnn_tensor = dnn.GetTensor();
      auto ddnn_tensor = ddnn.GetTensor();
      auto ntk_3_tensor = ntk_3.GetTensor();

      // Contraction old way
      Eigen::array<Eigen::IndexPair<int>, 0> tensor_product = {  };
      Eigen::array<Eigen::IndexPair<int>, 2> first_contraction = { Eigen::IndexPair<int>(2,2), Eigen::IndexPair<int>(3,5) };
      Eigen::Tensor<double, 6> d_mu_d_nu_f_ia = dnn_tensor.contract(dnn_tensor, tensor_product);
      Eigen::Tensor<double, 6> truth_tensor = ddnn_tensor.contract(d_mu_d_nu_f_ia, first_contraction);

      Eigen::Tensor<double, 0> guess = ntk_3_tensor.sum();
      Eigen::Tensor<double, 0> target = truth_tensor.sum();
      REQUIRE_THAT(guess(0), Catch::Matchers::WithinRel(target(0)));
    }
  }
  SUCCEED( " Observable test succeeded" );
}



TEST_CASE(" Test pointer to NNAD ") {
  const std::vector<int> arch{3, 5, 2};
  std::unique_ptr<NTK::NNAD> nn = std::make_unique<NTK::NNAD>(arch, 0, nnad::OutputFunction::QUADRATIC, false);
  int nout = arch.back();
  int np = nn->GetParameterNumber();
  int batch_size = 4;
  std::vector<NTK::data> data_vector (batch_size);
  data_vector[0] = {0.0, 0.3, 0.6};
  data_vector[1] = {0.0, 0.7, 0.9};
  data_vector[2] = {0.8, 0.1, 0.3};
  data_vector[3] = {0.1, 0.9, 0.5};

  NTK::dNN dnn (batch_size, nout, np);
  dnn.Evaluate(data_vector, nn.get());
  auto old_tensor = dnn.GetTensor();

  // Shift parameters
  double eps = 1.e-5;
  std::vector<double> new_parameters = nn->GetParameters();
  for (int ip = 0; ip < np; ip++)
    new_parameters[ip] = new_parameters[ip] * ( 1 + eps );

  nn->SetParameters(new_parameters);
  dnn.Evaluate(data_vector, nn.get());
  auto new_tensor = dnn.GetTensor();
  auto difference = new_tensor - old_tensor;
  Eigen::Tensor<double, 0> reduced_difference = difference.sum();
  REQUIRE_THAT(reduced_difference(0), !Catch::Matchers::WithinRel(0.));
  }
  