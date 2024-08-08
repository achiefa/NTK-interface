#include "NTK/Observable_impl.h"
#include "NTK/NumericalDerivative.h"

namespace NTK{
  const double eps = 1.e-5;

  //                        First derivative
  //________________________________________________________________________________
  dNN::dNN(int batch_size, int nout, int np) : BASIC<dNN, 3>(batch_size, nout, np) {}
  dNN::dNN(NNAD* nn, int batch_size) : BASIC<dNN, 3>(nn, batch_size, nn->GetArchitecture().back(), nn->GetParameterNumber()) {}
  dNN::dNN(NNAD* nn, std::vector<data> data_batch) : BASIC<dNN, 3>(data_batch, nn, data_batch.size(), nn->GetArchitecture().back(), nn->GetParameterNumber()) {}
  Tensor<2> dNN::algorithm_impl(const data &X, int a, NNAD* nn) {

    ////// Deprecated
    // std::vector<double> derivative = nn->Derive(X);
    // Eigen::TensorMap<Eigen::Tensor<double, 2, Eigen::ColMajor>> temp(derivative.data(), _d[1], _d[2] + 1); // Col-Major
    //
    // // Get rid of the first column (the outputs) and stores only first derivatives
    //  Eigen::array<Eigen::Index, 2> offsets = {0, 1};
    //  Eigen::array<Eigen::Index, 2> extents = {_d[1], _d[2]};
    //  return temp.slice(offsets, extents);
    ///////
    std::vector<double> derivative = helper::dNNAD_cleaner(nn,X);
    Eigen::TensorMap<Eigen::Tensor<double, 2, Eigen::ColMajor>> temp(derivative.data(), _d[1], _d[2]); // Col-Major
    return temp;
  }

  

  //                        Second derivative
  //________________________________________________________________________________________
  ddNN::ddNN(int batch_size, int nout, int np) : BASIC<ddNN, 4>(batch_size, nout, np, np) {}
  ddNN::ddNN(NNAD* nn, int batch_size) : BASIC<ddNN, 4>(nn, batch_size, nn->GetArchitecture().back(), nn->GetParameterNumber(), nn->GetParameterNumber()) {}
  ddNN::ddNN(NNAD* nn, std::vector<data> data_batch) : BASIC<ddNN, 4>(data_batch, nn, data_batch.size(), nn->GetArchitecture().back(), nn->GetParameterNumber(), nn->GetParameterNumber()) {}
  Tensor<3> ddNN::algorithm_impl(const data &X, int a, NNAD *nn) {

    ////// Deprecated
    //std::vector<double> results_vec = NTK::helper::HelperSecondFiniteDer(nn, X, eps); // Compute second derivatives
    //
    // // Store into ColMajor tensor
    // Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::ColMajor>> dd_NN(results_vec.data(), _d[1], _d[2] + 1, _d[3]);
    //
    // // reshape
    // Eigen::array<int, 3> new_shape{{0, 2, 1}};
    // Eigen::Tensor<double, 3> ddNN_reshape = dd_NN.shuffle(new_shape);
    //
    // // Get rid of the first column (the firs derivatives) and stores only second
    // Eigen::array<Eigen::Index, 3> offsets_3 = {0, 0, 1};
    // Eigen::array<Eigen::Index, 3> extents_3 = {_d[1], _d[2], _d[3]};
    // return ddNN_reshape.slice(offsets_3, extents_3); */
    //////

    std::vector<double> results_vec = NTK::helper::nd_dNNAD(nn, X, eps); // Compute second derivatives
    Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::ColMajor>> dd_NN(results_vec.data(), _d[1], _d[2], _d[3]);
    return dd_NN;
  }


  //                        Third derivative
  //________________________________________________________________________________________
  d3NN::d3NN(int batch_size, int nout, int np) : BASIC<d3NN, 5>(batch_size, nout, np, np, np) {}
  d3NN::d3NN(NNAD* nn, int batch_size) : BASIC<d3NN, 5>(nn,
                                                        batch_size,
                                                        nn->GetArchitecture().back(),
                                                        nn->GetParameterNumber(),
                                                        nn->GetParameterNumber(),
                                                        nn->GetParameterNumber()) {}
  d3NN::d3NN(NNAD* nn, std::vector<data> data_batch) : BASIC<d3NN, 5>(data_batch,
                                                                      nn,
                                                                      data_batch.size(),
                                                                      nn->GetArchitecture().back(),
                                                                      nn->GetParameterNumber(),
                                                                      nn->GetParameterNumber(),
                                                                      nn->GetParameterNumber()) {}
  Tensor<4> d3NN::algorithm_impl(const data &X, int a, NNAD *nn) {
    std::vector<double> third_der = NTK::helper::ndd_dNNAD(nn, X, eps); // Compute second derivatives

    // Store into ColMajor tensor
    Eigen::TensorMap<Eigen::Tensor<double, 4, Eigen::ColMajor>> d3_NN(third_der.data(), _d[1], _d[2], _d[3], _d[4]);
    Tensor<4> res = d3_NN;
    return res;
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
  /**
   * @brief
   *
   * The observable is computed as follows
   *
   *    O3_{i1 a1}{i2 a2}{i3 a3} = d_mu1 f_{i3 a3} * ( d_mu1_mu2 f_{i1 a1} d_mu2 f_{i2 a2} + d_mu1_mu2 f_{i2 a2} d_mu2 f_{i1 a1})
   *
   * and the order of the indices in the tensor is [ a3 ,i3 , a1, i1, a2, i2 ] ????????????/
   * 
   * @param dnn
   * @param ddnn
   * @return Tensor<6> 
   */
  Tensor<6> O3::contract_impl(dNN *dnn, ddNN *ddnn) {
    auto dnn_tensor = dnn->GetTensor(); // [ a, i, mu ]
    Tensor<4> ddnn_tensor = ddnn->GetTensor(); // [ a, i, mu1, mu2 ]

    // d_mu1_mu2 f_1 d_mu2 f_2
    Eigen::array<Eigen::IndexPair<int>, 1> ddNN_dnn_rule = {
      Eigen::IndexPair<int>(3, 2)
    };
    Tensor<5> ddNN_dnn = ddnn_tensor.contract(dnn_tensor, ddNN_dnn_rule); // [a1, i1, mu, a2, i2]

    // d_mu1_mu2 f_2 d_mu2 f_1
    Eigen::array<int, 5> transpose({3, 4, 2, 0, 1});
    Tensor<5> ddNN_dnn_symm = ddNN_dnn + ddNN_dnn.shuffle(transpose); // [a1, i1, mu, a2, i2] = [a2, i2, mu, a1, i1]

    // Contraction with d_mu1 f_3
    Eigen::array<Eigen::IndexPair<int>, 1> dnn_ddnn_dnn_rule = {
      Eigen::IndexPair<int>(2, 2)
    };
    Tensor<6> result = ddNN_dnn_symm.contract(dnn_tensor, dnn_ddnn_dnn_rule); // [a1, i1, a2, i2, a3, i3] = [a2, i2, a1, i1, a3, i3]

    return result;
  }

  //                                      O4
  //________________________________________________________________________________________
  O4::O4(int batch_size, int nout) : COMBINED<O4, 8>(batch_size, batch_size, batch_size, batch_size, nout, nout, nout, nout) {}
  /**
   * @brief Implementation
   * 
   * - I then symmetrise with ( 1 <-> 2 )
   * 
   * 
   * @param dnn 
   * @param d2nn 
   * @param d3nn 
   * @return Tensor<8> 
   */
  Tensor<8> O4::contract_impl(dNN *dnn, ddNN *d2nn, d3NN *d3nn) {
    auto dnn_tensor = dnn->GetTensor(); // [a, i, mu]
    auto d2nn_tensor = d2nn->GetTensor(); // [a, i, mu, nu]
    auto d3nn_tensor = d3nn->GetTensor(); // [a, i, mu, nu, rho]

    // part 1
    Eigen::array<Eigen::IndexPair<int>, 1> contraction_1 = {Eigen::IndexPair<int>(4, 2)};
    Eigen::array<Eigen::IndexPair<int>, 1> contraction_2 = {Eigen::IndexPair<int>(3, 3)};
    Eigen::array<Eigen::IndexPair<int>, 1> contraction_3 = {Eigen::IndexPair<int>(3, 2)};
    Eigen::array<Eigen::IndexPair<int>, 1> contraction_4 = {Eigen::IndexPair<int>(2, 3)};

    Eigen::array<int, 6> match_order({0, 1, 5, 2, 3, 4});
    auto ddd_f1_d_f2 = d3nn_tensor.contract(dnn_tensor, contraction_1); // [a1, i1, mu1, mu2, a2, i2]
    auto dd_f1_dd_f2 = (d2nn_tensor.contract(d2nn_tensor, contraction_2)).shuffle(match_order); // [a1, i1, mu2, a2, i2, mu1] -> [a1, i1, mu1, mu2, a2, i2]
    auto part1 = (ddd_f1_d_f2 + dd_f1_dd_f2).contract(dnn_tensor, contraction_3); // [a1, i1, mu1, a2, i2, a3, i3]

    auto dd_f1_d_f2 = d2nn_tensor.contract(dnn_tensor, contraction_3); // [a1, i1, mu2, a2, i2]
    auto part2 = dd_f1_d_f2.contract(d2nn_tensor, contraction_4); // [a1, i1, a2, i2, a3, i3, mu1]

    Eigen::array<Eigen::IndexPair<int>, 1> contraction_part1 = {Eigen::IndexPair<int>(2, 2)};
    Eigen::array<Eigen::IndexPair<int>, 1> contraction_part2 = {Eigen::IndexPair<int>(6, 2)};
    auto asy_result = part1.contract(dnn_tensor, contraction_part1) + part2.contract(dnn_tensor, contraction_part2); // [a1, i1, a2, i2, a3, i3, a4, i4]

    Eigen::array<int, 8> symmetrised({2, 3, 0, 1, 4, 5, 6, 7});
    Tensor<8> sym_result = asy_result + asy_result.shuffle(symmetrised);

    return sym_result;
  }

}