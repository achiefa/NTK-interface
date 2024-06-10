#include "NTK/Observable.h"
#include "NTK/NumericalDerivative.h"

namespace NTK{
  const double eps = 1.e-5;

  //                        First derivative
  //________________________________________________________________________________
  dNN::dNN(int batch_size, int nout, int np) : BASIC<dNN, 3>(batch_size, nout, np) {}
  dNN::dNN(NNAD* nn, int batch_size) : BASIC<dNN, 3>(nn, batch_size, nn->GetArchitecture().back(), nn->GetParameterNumber()) {}
  dNN::dNN(NNAD* nn, std::vector<data> data_batch) : BASIC<dNN, 3>(data_batch, nn, data_batch.size(), nn->GetArchitecture().back(), nn->GetParameterNumber()) {}
  Tensor<2> dNN::algorithm_impl(const data &X, int a, NNAD* nn) {
    std::vector<double> derivative = nn->Derive(X);
    Eigen::TensorMap<Eigen::Tensor<double, 2, Eigen::ColMajor>> temp(derivative.data(), _d[1], _d[2] + 1); // Col-Major

    // Get rid of the first column (the outputs) and stores only first derivatives
    Eigen::array<Eigen::Index, 2> offsets = {0, 1};
    Eigen::array<Eigen::Index, 2> extents = {_d[1], _d[2]};
    return temp.slice(offsets, extents);
  }

  

  //                        Second derivative
  //________________________________________________________________________________________
  ddNN::ddNN(int batch_size, int nout, int np) : BASIC<ddNN, 4>(batch_size, nout, np, np) {}
  ddNN::ddNN(NNAD* nn, int batch_size) : BASIC<ddNN, 4>(nn, batch_size, nn->GetArchitecture().back(), nn->GetParameterNumber(), nn->GetParameterNumber()) {}
  ddNN::ddNN(NNAD* nn, std::vector<data> data_batch) : BASIC<ddNN, 4>(data_batch, nn, data_batch.size(), nn->GetArchitecture().back(), nn->GetParameterNumber(), nn->GetParameterNumber()) {}
  Tensor<3> ddNN::algorithm_impl(const data &X, int a, NNAD *nn) {
    std::vector<double> results_vec = NTK::helper::HelperSecondFiniteDer(nn, X, eps); // Compute second derivatives

    // Store into ColMajor tensor
    Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::ColMajor>> dd_NN(results_vec.data(), _d[1], _d[2] + 1, _d[3]);

    // Swap to ColMajor for compatibility and reshape
    Eigen::array<int, 3> new_shape{{0, 2, 1}};
    Eigen::Tensor<double, 3> ddNN_reshape = dd_NN.shuffle(new_shape);

    // Get rid of the first column (the firs derivatives) and stores only second
    Eigen::array<Eigen::Index, 3> offsets_3 = {0, 0, 1};
    Eigen::array<Eigen::Index, 3> extents_3 = {_d[1], _d[2], _d[3]};
    return ddNN_reshape.slice(offsets_3, extents_3);
  }


  //                                      NTK
  //________________________________________________________________________________________
  O2::O2(int batch_size, int nout) : COMBINED<O2, 4>(batch_size, batch_size, nout, nout) {}
  Tensor<4> O2::contract_impl(dNN *dnn) {
    Eigen::array<Eigen::IndexPair<int>, 1> double_contraction = {Eigen::IndexPair<int>(2, 2)};
    auto dnn_tensor = dnn->GetTensor();
    return dnn_tensor.contract(dnn_tensor, double_contraction);
  }


  //                                      O3
  //________________________________________________________________________________________
  O3::O3(int batch_size, int nout) : COMBINED<O3, 6>(batch_size, batch_size, batch_size, nout, nout, nout) {}
  Tensor<6> O3::contract_impl(dNN *dnn, ddNN *ddnn) {
    auto dnn_tensor = dnn->GetTensor();
    auto ddnn_tensor = ddnn->GetTensor();
    Eigen::array<Eigen::IndexPair<int>, 0> tensor_product = {};
    Eigen::array<Eigen::IndexPair<int>, 2> first_contraction = {
        Eigen::IndexPair<int>(2, 2), Eigen::IndexPair<int>(3, 5)};
    auto d_mu_d_nu_f_ia = dnn_tensor.contract(dnn_tensor, tensor_product);
    return ddnn_tensor.contract(d_mu_d_nu_f_ia, first_contraction);
  }

}