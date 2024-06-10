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

    // reshape
    Eigen::array<int, 3> new_shape{{0, 2, 1}};
    Eigen::Tensor<double, 3> ddNN_reshape = dd_NN.shuffle(new_shape);

    // Get rid of the first column (the firs derivatives) and stores only second
    Eigen::array<Eigen::Index, 3> offsets_3 = {0, 0, 1};
    Eigen::array<Eigen::Index, 3> extents_3 = {_d[1], _d[2], _d[3]};
    return ddNN_reshape.slice(offsets_3, extents_3);
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
    std::vector<double> third_der = NTK::helper::HelperThirdFiniteDer(nn, X, eps); // Compute second derivatives

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
   *    O3_{i1 a1}{i2 a2}{i3 a3} = d_mu f_{i1 a1} d_nu f_{i2 a2} d_mu_nu f_{i3 a3} + (1 <-> 2)
   *
   * and the order of the indices in the tensor is [ a3 ,i3 , a1, i1, a2, i2 ].
   * 
   * @param dnn
   * @param ddnn
   * @return Tensor<6> 
   */
  Tensor<6> O3::contract_impl(dNN *dnn, ddNN *ddnn) {
    auto dnn_tensor = dnn->GetTensor();
    auto ddnn_tensor = ddnn->GetTensor();
    Eigen::array<Eigen::IndexPair<int>, 0> tensor_product = {};
    Eigen::array<Eigen::IndexPair<int>, 2> first_contraction = {
        Eigen::IndexPair<int>(2, 2), Eigen::IndexPair<int>(3, 5)};
    auto d_mu_d_nu_f_ia = dnn_tensor.contract(dnn_tensor, tensor_product);
    auto NTK_3 = ddnn_tensor.contract(d_mu_d_nu_f_ia, first_contraction);

    // Symmetrised
    Eigen::array<int, 6> transpose({0,1, 4, 5, 2, 3});
    auto NTK_3_symm = NTK_3 + NTK_3.shuffle(transpose);
    return NTK_3_symm;
  }

  //                                      O4
  //________________________________________________________________________________________
  O4::O4(int batch_size, int nout) : COMBINED<O4, 8>(batch_size, batch_size, batch_size, batch_size, nout, nout, nout, nout) {}
  /**
   * @brief Implementation
   * 
   * The observable is computed as follows
   * 
   * O4_{i1 a1}...{i4 a4} = d_mu1 ( d_mu_1_mu_2 f_{i3 a3} d_mu_2 f_{i1 a1} d_mu_3 f_{i2 a1} + ( 1 <-> 2 ) ) d_mu_1 f_{i4 a4} .
   * 
   * The expression above evaluates as
   * 
   * O4_{i1 a1}...{i4 a4} = ( |1| + |2| + |3| )_{i1 a1}...{i4 a4} + ( 1 <-> 2 ) ,
   * 
   * where
   * 
   * - |1| = d_mu_1...3 f_{i3 a3} d_mu_2 f_{i1 a1} d_mu_3 f_{i2 a2} d_mu_1 f_{i4 a4}
   *                                               ---------------------------------
   * - |2| = d_mu_2_3 f_{i3 a3} d_mu_1_2 f_{i1 a1} d_mu_3 f_{i2 a2} d_mu_1 f_{i4 a4}
   *                                               ---------------------------------
   * - |3| from |2| with ( 1 <-> 2 )
   * 
   * Note that the underlined part is common to |1| and |2|. I will proceed as follows:
   * - First, I compute the common object.
   * - I then compute |1| and |2|
   * - I obtain |3| from |2|
   * - I compute the first part of O4
   * - I then symmetrised with ( 1 <-> 2 )
   * 
   * 
   * @param dnn 
   * @param d2nn 
   * @param d3nn 
   * @return Tensor<8> 
   */
  Tensor<8> O4::contract_impl(dNN *dnn, ddNN *d2nn, d3NN *d3nn) {
    auto dnn_tensor = dnn->GetTensor();
    auto d2nn_tensor = d2nn->GetTensor();
    auto d3nn_tensor = d3nn->GetTensor();
    Eigen::array<Eigen::IndexPair<int>, 0> tensor_product = {};
    auto common_tensor = dnn_tensor.contract(dnn_tensor, tensor_product); // [ a2, i2, mu3, a4, i4, mu1]

    // Tensor 1
    Eigen::array<Eigen::IndexPair<int>, 3> contraction_1 = {
        Eigen::IndexPair<int>(2, 8), Eigen::IndexPair<int>(3, 2), Eigen::IndexPair<int>(4, 5) };
    auto tensor_1 = d3nn_tensor.contract( dnn_tensor.contract(common_tensor, tensor_product), contraction_1); // [ a3 i3 a1 i1 a2 i2 a4 i4 ]

    // Tensor 2
    Eigen::array<Eigen::IndexPair<int>, 1> contraction_1_1 = {Eigen::IndexPair<int>(2,5)}; // [ a1 i1 mu2 a2 i2 mu3 a4 i4 ]
    Eigen::array<Eigen::IndexPair<int>, 2> contraction_1_2= {Eigen::IndexPair<int>(2,2), Eigen::IndexPair<int>(3, 5)};
    auto tensor_2 = d2nn_tensor.contract( d2nn_tensor.contract(common_tensor, contraction_1_1),   contraction_1_2); // [ a3 i3 a1 i1 a2 i2 a4 i4 ]

    // O4_asy
    Eigen::array<int, 8> transpose({0, 1, 4, 5, 2, 3, 6, 7});
    auto O4_asy = tensor_1 + tensor_2 + tensor_2.shuffle(transpose);

    Tensor<8> O4_sym = O4_asy + O4_asy.shuffle(transpose);
    return O4_sym;
  }

}