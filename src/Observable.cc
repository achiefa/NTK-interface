#include "NTK/Observable.h"

namespace NTK{

  dNN::dNN(int size1, int size2, int size3) : BASIC<dNN, 3>(size1, size2, size3) {}
  Tensor<2> dNN::algorithm_impl(const data &X, int a, NNAD* nn) {
    // .data() is needed because returns a direct pointer to the memory array
    // used internally by the vector
    Eigen::TensorMap<Eigen::Tensor<double, 2, Eigen::ColMajor>> temp(
        nn->Derive(X).data(), _d[1], _d[2] + 1); // Col-Major

    // Get rid of the first column (the outputs) and stores only first
    // derivatives
    Eigen::array<Eigen::Index, 2> offsets = {0, 1};
    Eigen::array<Eigen::Index, 2> extents = {_d[1], _d[2]};
    return temp.slice(offsets, extents);
  }
}