#include "NTK/NumericalDerivative.h"
#include "NTK/Observable.h"
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

double sum_utility_return_dNN(int batch_size, int nout, int np, NTK::NNAD* nn, std::vector<NTK::data> vec_data) {
  auto tensor = utility_return_dNN(batch_size, nout, np, nn, vec_data);
  Eigen::Tensor<double, 0> target = tensor.sum();
  return target(0);
}

const double eps = 1.e-5;

NTK::Tensor<4> utility_return_ddNN(int batch_size, int nout, int np, NTK::NNAD* nn, std::vector<NTK::data> vec_data) {
  NTK::Tensor<4> dd_NN (batch_size, nout, np, np);
  dd_NN.setZero();

  for (int a = 0; a < batch_size; a++){
    std::vector<double> input_a = vec_data[a];
    std::vector<double> results_vec = NTK::helper::HelperSecondFiniteDer(nn, input_a, eps); // Compute second derivatives

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

double sum_utility_return_ddNN(int batch_size, int nout, int np, NTK::NNAD* nn, std::vector<NTK::data> vec_data) {
  auto tensor = utility_return_ddNN(batch_size, nout, np, nn, vec_data);
  Eigen::Tensor<double, 0> target = tensor.sum();
  return target(0);
}