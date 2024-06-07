#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include "NTK/Observable.h"
#include "NTK/NumericalDerivative.h"
#include "NNAD/FeedForwardNN.h"

NTK::Tensor<3> utility_return_dNN(int batch_size, int nout, int np, NTK::NNAD* nn, std::vector<NTK::data> vec_data) {
  Eigen::Tensor<double, 3> d_NN (batch_size, nout, np);
  d_NN.setZero();

  for (int a = 0; a < batch_size; a++){
    std::vector<double> input_a = vec_data[a];
    // -------------------------- First derivative -------------------------
    // .data() is needed because returns a direct pointer to the memory array used internally by the vector
    std::vector<double> DD = nn->Derive(input_a);
    Eigen::TensorMap< Eigen::Tensor<double, 2, Eigen::ColMajor> > temp (DD.data(), nout, np + 1); // Col-Major

    // Get rid of the first column (the outputs) and stores only first derivatives
    Eigen::array<Eigen::Index, 2> offsets = {0, 1};
    Eigen::array<Eigen::Index, 2> extents = {nout, np};
    d_NN.chip(a,0) = temp.slice(offsets, extents);
  }
  return d_NN;
}


//---------------------------------------------------------------------
//        TEST
//---------------------------------------------------------------------
TEST_CASE( " Testing dNN", "[Observable][dNN] " ) {
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
    std::unique_ptr<NTK::dNN> dnn = std::make_unique<NTK::dNN> (batch_size, nout, np);
    auto tensor = dnn->GetTensor();
    Eigen::Tensor<double, 0> result = tensor.sum();
    REQUIRE_FALSE(dnn == NULL);
    REQUIRE_THAT(result(0), Catch::Matchers::WithinRel(0.));
  }

  SECTION( " Testing  id ") {
    std::unique_ptr<NTK::dNN> dnn = std::make_unique<NTK::dNN> (batch_size, nout, np);
    std::string truth = "NTK::dNN";
    REQUIRE(NTK::dNN::id == truth);
    REQUIRE(dnn->GetID() == truth);
  }

  SECTION( " Testing evaluation , `data_map`, and`is_computed` " ) {
    std::unique_ptr<NTK::dNN> dnn = std::make_unique<NTK::dNN> (batch_size, nout, np);
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

    SECTION( "Testin against truth iterative" ) {
      auto truth = utility_return_dNN(batch_size, nout, np, nn.get(), data_vector);
      for (int a = 0; a < data_vector.size(); a++) {
        dnn->Evaluate(data_vector[a], a, nn.get());
      }
      Eigen::Tensor<double, 0> target = truth.sum();
      Eigen::Tensor<double, 0> guess = dnn->GetTensor().sum();
      REQUIRE_THAT(guess(0), Catch::Matchers::WithinRel(target(0)));
    } 

    SUCCEED( " Evaluation test succeeded" );
  }
  SUCCEED( " dNN test succeeded" );
}

