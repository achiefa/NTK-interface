#pragma once
#include "NTK/NumericalDerivative.h"
#include "NTK/Observable.h"
#include "NNAD/FeedForwardNN.h"
#include <list>


struct NonZeroElement {
  NonZeroElement(std::function<double(double)> f, std::vector<int> list) : _index(list), _f(f) {}
  std::vector<int> _index;
  std::function<double(double)> _f;
};

class KnownNNAD {
  public:
  KnownNNAD(nnad::FeedForwardNN<double> *NN) {
    b1 = NN->GetBias(1,0);
    b2 = NN->GetBias(1,1);
    B  = NN->GetBias(2,0);
    w1 = NN->GetLink(1, 0, 0);
    w2 = NN->GetLink(1, 1, 0);
    W1 = NN->GetLink(2, 0, 0);
    W2 = NN->GetLink(2, 0, 1);
    CreateMap();
  }

  static std::map<int,int> DummyToNNAD;


  double tanh(double x) { return nnad::Tanh<double> (x); }
  double dtanh(double x) { return nnad::dTanh<double> (x); }
  double ddtanh(double x) { return -2. * tanh(x) * dtanh(x); }
  double dddtanh(double x) { return -2. * dtanh(x) * dtanh(x) - 2. * tanh(x) * ddtanh(x); }
  double phi_1 (double x) { return b1 + w1 * x; }
  double phi_2 (double x) { return b2 + w2 * x; }
  double PHI (double x) { return B + W1 * tanh(phi_1 (x)) + W2 * tanh(phi_2 (x)); }

  double H13 (double x) { return dtanh(phi_1(x)); }
  double H15 (double x) { return dtanh(phi_1(x)) * x; }
  double H24 (double x) { return dtanh(phi_2(x)); }
  double H26 (double x) { return dtanh(phi_2(x)) * x; }
  double H33 (double x) { return W1 * ddtanh(phi_1(x)); }
  double H35 (double x) { return W1 * x * ddtanh(phi_1(x)); }
  double H44 (double x) { return W2 * ddtanh(phi_2(x)); }
  double H46 (double x) { return W2 * x * ddtanh(phi_2(x)); }
  double H55 (double x) { return W1 * ddtanh(phi_1(x)) * x * x; }
  double H66 (double x) { return W2 * ddtanh(phi_2(x)) * x * x; }

  std::function<double(double)> t133 { [&] (double x) -> double { return ddtanh(phi_1(x)); } };
  std::function<double(double)> t135 { [&] (double x) -> double { return ddtanh(phi_1(x)) * x; }};
  std::function<double(double)> t155 { [&] (double x) -> double { return ddtanh(phi_1(x)) * x * x; }};
  std::function<double(double)> t244 { [&] (double x) -> double { return ddtanh(phi_2(x)); }};
  std::function<double(double)> t246 { [&] (double x) -> double { return ddtanh(phi_2(x)) * x; }};
  std::function<double(double)> t266 { [&] (double x) -> double { return ddtanh(phi_2(x)) * x * x; }};
  std::function<double(double)> t333 { [&] (double x) -> double { return dddtanh(phi_1(x)) * W1; }};
  std::function<double(double)> t335 { [&] (double x) -> double { return dddtanh(phi_1(x)) * x * W1; }};
  std::function<double(double)> t355 { [&] (double x) -> double { return dddtanh(phi_1(x)) * x * x * W1; }};
  std::function<double(double)> t444 { [&] (double x) -> double { return dddtanh(phi_2(x)) * W2; }};
  std::function<double(double)> t446 { [&] (double x) -> double { return dddtanh(phi_2(x)) * x * W2; }};
  std::function<double(double)> t466 { [&] (double x) -> double { return dddtanh(phi_2(x)) * x * x * W2; }};
  std::function<double(double)> t555 { [&] (double x) -> double { return dddtanh(phi_1(x)) * x * x * x * W1; }};
  std::function<double(double)> t666 { [&] (double x) -> double { return dddtanh(phi_2(x)) * x * x * x * W2; }};
  

  void CreateMap(){
    std::vector<NonZeroElement> NonZeroValues;
    NonZeroValues.push_back( NonZeroElement (t133, {1,3,3}));
    NonZeroValues.push_back( NonZeroElement (t135, {1,3,5}));
    NonZeroValues.push_back( NonZeroElement (t155, {1,5,5}));
    NonZeroValues.push_back( NonZeroElement (t244, {2,4,4}));
    NonZeroValues.push_back( NonZeroElement (t246, {2,4,6}));
    NonZeroValues.push_back( NonZeroElement (t266, {2,6,6}));
    NonZeroValues.push_back( NonZeroElement (t333, {3,3,3}));
    NonZeroValues.push_back( NonZeroElement (t335, {3,3,5}));
    NonZeroValues.push_back( NonZeroElement (t355, {3,5,5}));
    NonZeroValues.push_back( NonZeroElement (t444, {4,4,4}));
    NonZeroValues.push_back( NonZeroElement (t446, {4,4,6}));
    NonZeroValues.push_back( NonZeroElement (t466, {4,6,6}));
    NonZeroValues.push_back( NonZeroElement (t555, {5,5,5}));
    NonZeroValues.push_back( NonZeroElement (t666, {6,6,6}));
    _NonZeroValues = NonZeroValues;
  }

  std::vector<double> jacobian(double x) {
    std::vector<double> jaco {1.0,
                              tanh(phi_1(x)),
                              tanh(phi_2(x)),
                              W1 * dtanh(phi_1(x)),
                              W2 * dtanh(phi_2(x)),
                              W1 * x * dtanh(phi_1(x)),
                              W2 * x * dtanh(phi_2(x))};
    return jaco;
  }

  Eigen::VectorXd Eigen_wrapper_jacobian(double x) {
    std::vector<double> jaco = jacobian(x);
    Eigen::Matrix<double,7 ,1> res {
      jaco[0],
      jaco[1],
      jaco[2],
      jaco[3],
      jaco[4],
      jaco[5],
      jaco[6]
    };
    return res;
  }

  Eigen::Tensor<double,1> Tensor_wrapper_jacobian(double x) {
    std::vector<double> jaco = jacobian(x);
    Eigen::TensorMap<Eigen::Tensor<double,1>> res(jaco.data(), jaco.size());
    return res;
  }


  Eigen::MatrixXd Hessian (double x) {
    Eigen::MatrixXd hes {
//     B    W1   W2   b1   b2   w1   w2
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},  // B 
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},  // W1
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},  // W2
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},  // b1
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},  // b2
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},  // w1
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}   // w2
      };

      hes(1,3) = H13(x);
      hes(3,1) = hes(1,3);
      hes(1,5) = H15(x);
      hes(5,1) = hes(1,5);
      hes(2,4) = H24(x);
      hes(4,2) = hes(2,4);
      hes(2,6) = H26(x);
      hes(6, 2) = hes(2, 6);
      hes(3,3) = H33(x);
      hes(3,5) = H35(x);
      hes(5,3) = hes(3,5);
      hes(4,4) = H44(x);
      hes(4,6) = H46(x);
      hes(6,4) = hes(4,6);
      hes(5,5) = H55(x);
      hes(6,6) = H66(x);

    return hes;
  }

  Eigen::Tensor<double, 2> tensor_wrapper_hessian(double x) {
    Eigen::MatrixXd hessian = Hessian(x);
    Eigen::TensorMap<Eigen::Tensor<double,2, Eigen::ColMajor>> result (hessian.data(), hessian.rows(), hessian.cols());
    return result;
  }

  Eigen::Tensor<double,3> ThirdDerivative (double x) {
    Eigen::Tensor<double, 3> Result (7, 7, 7);
    Result.setZero();

    for (auto& nzv : _NonZeroValues) {
      //std::cout << "_________________________" << std::endl;
      do {
        Result(nzv._index[0], nzv._index[1], nzv._index[2]) = nzv._f(x);
        //std::cout << " " << nzv._index[0] << "  " <<  nzv._index[1] << "  " << nzv._index[2] << std::endl;
      } while ( std::next_permutation(nzv._index.begin(), nzv._index.end()) );
    }
    return Result;
  }

  double NTK (double x1, double x2) {
    std::vector<double> jaco1 = jacobian(x1);
    std::vector<double> jaco2 = jacobian(x2);
    double res = std::inner_product(jaco1.begin(), jaco1.end(), jaco2.begin(), 0.);
    return res;
  }

  double O3 (double xa, double xb, double xg) {
    Eigen::MatrixXd Ha = Hessian(xa);
    Eigen::MatrixXd Hb = Hessian(xb);
    Eigen::VectorXd Ja = Eigen_wrapper_jacobian(xa);
    Eigen::VectorXd Jb = Eigen_wrapper_jacobian(xb);
    Eigen::VectorXd Jg = Eigen_wrapper_jacobian(xg);

    double res =  Jg.dot(Ha * Jb + Hb * Ja);
    return res;
  }

  double O4 (double x1, double x2, double x3, double x4) {
    auto j1 = Tensor_wrapper_jacobian(x1);
    auto j2 = Tensor_wrapper_jacobian(x2);
    auto j3 = Tensor_wrapper_jacobian(x3);
    auto j4 = Tensor_wrapper_jacobian(x4);
    auto H1 = tensor_wrapper_hessian(x1);
    auto H2 = tensor_wrapper_hessian(x2);
    auto H3 = tensor_wrapper_hessian(x3);
    auto T1 = ThirdDerivative(x1);
    auto T2 = ThirdDerivative(x2);

    // Part 1
    Eigen::array<Eigen::IndexPair<int>, 1> T_j = {Eigen::IndexPair<int>(2, 0)};
    Eigen::array<Eigen::IndexPair<int>, 1> H_H = {Eigen::IndexPair<int>(1, 1)};
    Eigen::array<Eigen::IndexPair<int>, 1> part1_rule = {Eigen::IndexPair<int>(1, 0)};
    auto temp = T1.contract(j2, T_j) +
                T2.contract(j1, T_j) +
                H1.contract(H2, H_H) +
                H2.contract(H1, H_H);
    Eigen::Tensor<double,1> part1 = temp.contract(j3, part1_rule);

    // Part
    Eigen::array<Eigen::IndexPair<int>, 1> H_j = {Eigen::IndexPair<int>(1, 0)};
    Eigen::array<Eigen::IndexPair<int>, 1> part2_rule = {Eigen::IndexPair<int>(1, 0)};
    auto temp2 = H1.contract(j2, H_j) +
                 H2.contract(j1, H_j);
    Eigen::Tensor<double, 1> part2 = temp2.contract(H3, part2_rule);

    // Last contraction
    Eigen::array<Eigen::IndexPair<int>, 1> last_contraction = {Eigen::IndexPair<int>(0, 0)};
    Eigen::Tensor<double,0> result = part1.contract(j4, last_contraction) + part2.contract(j4, last_contraction);

    return result(0);
  }

  private:
  double b1, b2, w1, w2, B, W1, W2;
  std::vector<NonZeroElement> _NonZeroValues;
};
std::map<int,int> KnownNNAD::DummyToNNAD {{0,0}, {1,1}, {2,2}, {3,3}, {4,5}, {5,4}, {6,6}};

std::function<std::vector<double> (std::vector<double> const&, std::vector<double>)> PolTestFunction
    {
      [] (std::vector<double> const& X, std::vector<double> parameters) -> std::vector<double>
      {
        double theta_1 = parameters[0];
        double theta_2 = parameters[1];
        double theta_3 = parameters[2];
        double x = X[0];
        std::vector<double> res{1. + theta_1 * x + pow(theta_2, 2) * pow(x, 3) + pow(theta_3, 4) * pow(x, 5)};
        return res;
      }
    };