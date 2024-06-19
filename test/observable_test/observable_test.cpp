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

  SECTION( "Testing constructor " ) {
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

    SECTION( " Testing `is_copmuted` method ") {
      REQUIRE_FALSE(dnn->is_computed());
      dnn->Evaluate(data_vector , nn.get());
      REQUIRE(dnn->is_computed());
    }
  }
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

  SECTION( " Testing against dummy NN ") {
    const std::vector<int> arch{1, 2, 1};
    std::unique_ptr<NTK::NNAD> nn = std::make_unique<NTK::NNAD>(arch, 0, false,
                                        nnad::Tanh<double>,  nnad::dTanh<double>,
                                        nnad::OutputFunction::LINEAR);
    KnownNNAD dummyNN(nn.get());
    NTK::data xa {0.5};
    NTK::data xb {0.2};
    NTK::data xg {0.7};
    NTK::data xd {0.1};
    const int nout = 1;

    SECTION( " Testing NTK ") {
      NTK::dNN dnn(nn.get(), {xa, xb});
      NTK::O2 ntk (2, nout);
      dnn.Evaluate();
      ntk.Evaluate(&dnn);
      auto ntk_tensor = ntk.GetTensor();
      double NTK_aa = dummyNN.NTK(xa[0], xa[0]);
      double NTK_ab = dummyNN.NTK(xa[0], xb[0]);
      double NTK_ba = dummyNN.NTK(xb[0], xa[0]);
      double NTK_bb = dummyNN.NTK(xb[0], xb[0]);
      REQUIRE_THAT(NTK_aa, Catch::Matchers::WithinRel(ntk_tensor(0,0,0,0)));
      REQUIRE_THAT(NTK_ab, Catch::Matchers::WithinRel(ntk_tensor(0,1,0,0)));
      REQUIRE_THAT(NTK_ba, Catch::Matchers::WithinRel(ntk_tensor(1,0,0,0)));
      REQUIRE_THAT(NTK_bb, Catch::Matchers::WithinRel(ntk_tensor(1,1,0,0)));
    }

    SECTION( " Testing O3 ") {
      std::vector<NTK::data> batch {xa,xb,xg};
      NTK::dNN dnn(nn.get(), {xa, xb, xg});
      NTK::ddNN ddnn(nn.get(), {xa, xb, xg});
      NTK::O3 O3 (3, nout);
      dnn.Evaluate();
      ddnn.Evaluate();
      O3.Evaluate(&dnn, &ddnn);
      auto O3_tensor = O3.GetTensor();
      double O3_known;

      // Test all possible combinations
      for(int i1=0; i1 < 3; i1++) {
        for(int i2=0; i2 < 3; i2++) {
          for(int i3=0; i3 < 3; i3++) {
            O3_known = dummyNN.O3(batch[i1][0], batch[i2][0], batch[i3][0]);
            REQUIRE_THAT(O3_known, Catch::Matchers::WithinAbs(O3_tensor(i1, 0, i2, 0, i3, 0), 1.e-6));
          }
        }
      }
    }

    SECTION(" Testing O4 ") {
      std::vector<NTK::data> batch {xa, xb, xg, xd};
      NTK::dNN dnn(nn.get(), batch);
      NTK::ddNN ddnn(nn.get(), batch);
      NTK::d3NN d3nn(nn.get(), batch);
      NTK::O4 O4 (4, nout);
      dnn.Evaluate();
      ddnn.Evaluate();
      d3nn.Evaluate();
      O4.Evaluate(&dnn, &ddnn, &d3nn);
      auto O4_tensor = O4.GetTensor();
      double O4_known;

      // Test all possible combinations
      for(int i1=0; i1 < 4; i1++) {
        for(int i2=0; i2 < 4; i2++) {
          for(int i3=0; i3 < 4; i3++) {
            for(int i4=0; i4 < 4; i4++) {
              O4_known = dummyNN.O4(batch[i1][0], batch[i2][0], batch[i3][0], batch[i4][0]);
              REQUIRE_THAT(O4_known, Catch::Matchers::WithinAbs(O4_tensor(i1, 0, i2, 0, i3, 0, i4, 0), 1.e-4));
            }
          }
        }
      }
    }
  }
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
  