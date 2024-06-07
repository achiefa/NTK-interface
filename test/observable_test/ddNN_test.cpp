#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include "NTK/Observable.h"
#include "NTK/NumericalDerivative.h"
#include "NNAD/FeedForwardNN.h"

const double eps = 1.e-5;

NTK::Tensor<4> utility_return_ddNN(int batch_size, int nout, int np, NTK::NNAD* nn, std::vector<NTK::data> vec_data) {
  NTK::Tensor<4> dd_NN (batch_size, nout, np, np);
  dd_NN.setZero();

  for (int a = 0; a < batch_size; a++){
    std::vector<double> input_a = vec_data[a];
    std::vector<double> results_vec = NTK::helper::HelperSecondFiniteDer(nn, nn->GetParameters(), input_a, np, nout, eps); // Compute second derivatives

    // Store into ColMajor tensor
    // The order of the dimensions has been tested in "SecondDerivative", and worked out by hand.
    Eigen::TensorMap< Eigen::Tensor<double, 3, Eigen::ColMajor> > ddNN (results_vec.data(), nout, np + 1, np);

    // Swap to ColMajor for compatibility and reshape
    Eigen::array<int, 3> new_shape{{0, 2, 1}};
    Eigen::Tensor<double, 3> ddNN_reshape = ddNN.shuffle(new_shape);

    // Get rid of the first column (the firs derivatives) and stores only second derivatives
    Eigen::array<Eigen::Index, 3> offsets_3 = {0, 0, 1};
    Eigen::array<Eigen::Index, 3> extents_3 = {nout, np, np};
    dd_NN.chip(a,0) = ddNN_reshape.slice(offsets_3, extents_3);
  }
  return dd_NN;
}

//---------------------------------------------------------------------
//        TESTS
//---------------------------------------------------------------------
TEST_CASE( " Testing ddNN", "[Observable][ddNN] " ) {
  INFO("Initialising neural network and data");
  const std::vector<int> arch{3, 5, 2};
  std::unique_ptr<NTK::NNAD> nn = std::make_unique<NTK::NNAD>(arch, 0, nnad::OutputFunction::QUADRATIC, false);
  NTK::NNAD* nn_ptr = new NTK::NNAD(arch, 0, nnad::OutputFunction::QUADRATIC, false);
  int nout = arch.back();
  int np = nn->GetParameterNumber();
  int batch_size = 4;
  std::vector<NTK::data> data_vector (batch_size);
  data_vector[0] = {0.0, 0.3, 0.6};
  data_vector[1] = {0.0, 0.7, 0.9};
  data_vector[2] = {0.8, 0.1, 0.3};
  data_vector[3] = {0.1, 0.9, 0.5};

  SECTION( "Testing constructor" ) {
    std::unique_ptr<NTK::ddNN> ddnn = std::make_unique<NTK::ddNN> (batch_size, nout, np);
    auto tensor = ddnn->GetTensor();
    Eigen::Tensor<double, 0> result = tensor.sum();
    REQUIRE_FALSE(ddnn == NULL);
    REQUIRE_THAT(result(0), Catch::Matchers::WithinRel(0.));
  }

  SECTION( " Testing  id ") {
    std::unique_ptr<NTK::ddNN> ddnn = std::make_unique<NTK::ddNN> (batch_size, nout, np);
    std::string truth = "NTK::ddNN";
    REQUIRE(NTK::ddNN::id == truth);
    REQUIRE(ddnn->GetID() == truth);
  }

  SECTION( " Testing evaluation , `data_map`, and`is_computed` " ) {
    std::unique_ptr<NTK::ddNN> ddnn = std::make_unique<NTK::ddNN> (batch_size, nout, np);
    int i = 3;

    SECTION( " Testing Evaluation ") {
      REQUIRE_NOTHROW(ddnn->Evaluate(data_vector[i], i , nn.get()));
      REQUIRE_NOTHROW(ddnn->Evaluate(data_vector, nn.get()));
    }

    SECTION( " Testing data_map ") {
      INFO("Checking if data_map is empty before evaluation");
      REQUIRE(ddnn->GetDataMap().empty());

      INFO("Checking if data_map is empty after single evaluation");
      ddnn->Evaluate(data_vector[i], i , nn.get());
      REQUIRE_FALSE(ddnn->GetDataMap().empty());

      INFO("Check map has been indexed correcly");
      REQUIRE(ddnn->GetDataMap().count(i) == 1);
      REQUIRE(ddnn->GetDataMap().count(0) == 0);
      REQUIRE(ddnn->GetDataMap().count(1) == 0);
      REQUIRE(ddnn->GetDataMap().count(2) == 0);
      REQUIRE(ddnn->GetDataMap().size() == 1);
    }

    SECTION( " Testing `is_copmuted` ") {
      REQUIRE_FALSE(ddnn->is_computed());
      ddnn->Evaluate(data_vector , nn.get());
      REQUIRE(ddnn->is_computed());
    }

    SECTION( "Testin against truth iterative" ) {
      auto truth = utility_return_ddNN(batch_size, nout, np, nn.get(), data_vector);
      for (int a = 0; a < data_vector.size(); a++) {
        ddnn->Evaluate(data_vector[a], a, nn.get());
      }
      Eigen::Tensor<double, 0> target = truth.sum();
      Eigen::Tensor<double, 0> guess = ddnn->GetTensor().sum();
      REQUIRE_THAT(guess(0), Catch::Matchers::WithinRel(target(0)));
    } 

    SUCCEED( " Evaluation test succeeded" );
  }
  SUCCEED( " ddNN test succeeded" );
}

