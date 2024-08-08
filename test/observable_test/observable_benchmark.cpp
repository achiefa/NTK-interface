#include "NTK/Observable.h"
#include "NNAD/FeedForwardNN.h"
#include "NTK/utility_test.hpp"

#include "benchmark/benchmark.h"

//____________________________________________________________________
void O2_O3_old(benchmark::State &s) {

  // Size of the network
  int N = s.range(0);
  const std::vector<int> arch{1, N, N, 1};
  const double eps = 1.e-5;

  // Initialise network
  std::unique_ptr<NTK::NNAD> nn = std::make_unique<NTK::NNAD>(arch, 0, nnad::OutputFunction::QUADRATIC, false);
  int nout = arch.back();
  int np = nn->GetParameterNumber();

  // Initialise data batch
  int batch_size = 5;
  std::vector<NTK::data> data_vector (batch_size);
  data_vector[0] = {0.0};
  data_vector[1] = {0.25};
  data_vector[2] = {0.50};
  data_vector[3] = {0.75};
  data_vector[4] = {1}; 

  // Compute the NTK in the old way
  Eigen::Tensor<double, 3> d_NN (batch_size, nout, np);
  Eigen::Tensor<double, 4> dd_NN (batch_size, nout, np, np);
  d_NN.setZero();
  dd_NN.setZero();

    for (int a = 0; a < batch_size; a++){
      std::vector<double> input_a = data_vector[a];

      // -------------------------- First derivative -------------------------
      // .data() is needed because returns a direct pointer to the memory array used internally by the vector
      std::vector<double> DD = nn->Derive(input_a);
      Eigen::TensorMap< Eigen::Tensor<double, 2, Eigen::ColMajor> > temp (DD.data(), nout, np + 1); // Col-Major

      // Get rid of the first column (the outputs) and stores only first derivatives
      Eigen::array<Eigen::Index, 2> offsets = {0, 1};
      Eigen::array<Eigen::Index, 2> extents = {nout, np};
      d_NN.chip(a,0) = temp.slice(offsets, extents);


      // -------------------------- Second derivative -------------------------
      std::vector<double> results_vec = NTK::helper::HelperSecondFiniteDer(nn.get(), input_a, eps); // Compute second derivatives

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
    // Contract first derivatives to get the NTK
    Eigen::array<Eigen::IndexPair<int>, 1> double_contraction = { Eigen::IndexPair<int>(2,2) };
    Eigen::Tensor<double, 4> NTK_Eigen = d_NN.contract(d_NN, double_contraction);

    // Contract first and second derivatives
    Eigen::array<Eigen::IndexPair<int>, 0> tensor_product = {  };
    Eigen::array<Eigen::IndexPair<int>, 2> first_contraction = { Eigen::IndexPair<int>(2,2), Eigen::IndexPair<int>(3,5) };
    Eigen::Tensor<double, 6> d_mu_d_nu_f_ia = d_NN.contract(d_NN, tensor_product);
    Eigen::Tensor<double, 6> O3 = dd_NN.contract(d_mu_d_nu_f_ia, first_contraction);
    Eigen::array<int, 6> transpose({0,1, 4, 5, 2, 3});
    Eigen::Tensor<double, 6> O3_symm = O3 + O3.shuffle(transpose);
}


//____________________________________________________________________
void O2_O3_new(benchmark::State &s) {

  // Size of the network
  int N = s.range(0);
  const std::vector<int> arch{1, N, N, 1};

  // Initialise network
  std::unique_ptr<NTK::NNAD> nn = std::make_unique<NTK::NNAD>(arch, 0, nnad::OutputFunction::QUADRATIC, false);
  int nout = arch.back();
  int np = nn->GetParameterNumber();

  // Initialise data batch
  int batch_size = 5;
  std::vector<NTK::data> data_vector (batch_size);
  data_vector[0] = {0.0};
  data_vector[1] = {0.25};
  data_vector[2] = {0.50};
  data_vector[3] = {0.75};
  data_vector[4] = {1}; 

  // Compute the NTK in the new way
  NTK::dNN dnn(nn.get(), data_vector);
  NTK::ddNN ddnn(nn.get(), data_vector);
  NTK::O2 o2(nout, np);
  NTK::O3 o3(nout, np);
  dnn.Evaluate();
  ddnn.Evaluate();
  o2.Evaluate(&dnn);
  o3.Evaluate(&dnn, &ddnn);
}

BENCHMARK(O2_O3_old)
  ->Arg(5)
  ->Arg(10)
  ->Arg(20)
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

BENCHMARK(O2_O3_new)
  ->Arg(5)
  ->Arg(10)
  ->Arg(20)
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

BENCHMARK_MAIN();