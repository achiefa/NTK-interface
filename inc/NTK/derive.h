// NNAD

#include <fstream>
#include <functional>
#include <algorithm>
#include <iostream>

namespace NTK
{
  std::vector<std::vector<double>> FiniteDifference (
    std::function<std::vector<double> (std::vector<double> const&, std::vector<double>)> f,
    std::vector<double> parameters, 
    std::vector<double> const& x, 
    double const& eps)
  {
    // TODO
    std::vector<std::vector<double>> results;
    int np = int(parameters.size());

    for (int ip = 0; ip < np; ip++) {
      std::vector<double> ParForward = parameters;
      std::vector<double> ParBackward = parameters;

      ParForward[ip] = parameters[ip] * ( 1 + eps );
      ParBackward[ip] = parameters[ip] * ( 1 - eps );

      std::vector<double> fp = f(x, ParForward);
      std::vector<double> fm = f(x, ParBackward);
      std::vector<double> result (fp.size(), 0.);
      std::transform(fp.begin(), fp.end(), fm.begin(), result.begin(), [&] (double fplus, double fminus) 
      {
        return 0.5 * (fplus - fminus) / eps / parameters[ip];
      });

      // TODO
      // The naming is very bad. Because of the behaviour of std::transform,
      // fm is modified and eventually coincides with the derivative. Still,
      // the name is confusing.
      results.push_back(result);
    }
    return results;
  }

  // This is the same function as before, but returns the derivatives as
  // a single vector object. The ordering of the vector is as follows:
  //
  //    results_jk = results[ j + k * dim(f) ]   Col-Major
  //
  // where
  //      j = 0, ..., dim(f) - 1 and k = 0, ..., dim(np) - 1.
  //
  // It may happen that the function "f" is already a vectorial representation
  // of some tensor. For instance, f can represent the vector of first order
  // derivatives of a NN (and evaluations from the back-propagation). In this
  // case, f is a vector that represents a two-rank tensor, where the indices
  // run over the outputs and the parameters (remember that in the case of NNAD,
  // the method `derive` also provides the evaluation of the NN). Then, the
  // the actual dimension of f is the products of the two sub-dimensions that
  // build up f, namely dim(f) = dim(f_1) * dim(f_2). Moreover, also the index
  // linked to f can actually be splitted into two separates indices that run
  // over the two dimensions of f respectively. Assuming also f is in col-major
  // order, we have j -> i + j * dim(f_1), where
  //            i = 0, ..., dim(f_1) - 1  and  j = 0, ..., dim(f_2) - 1,
  // and the 3-rank tensor can then be written as
  //
  //     results_ijk = results[ i + j * dim(f_1) + k * dim(f) * dim(f_2)]   Col-Major
  std::vector<double> FiniteDifferenceVec (
    std::function<std::vector<double> (std::vector<double> const&, std::vector<double>)> f,
    std::vector<double> parameters,
    std::vector<double> const& x,
    int const& size_f,
    double const& eps)
  {
    int np = int(parameters.size());
    std::vector<double> results (np * size_f, 0.);
    int Counter = 0;

    for (int ip = 0; ip < np; ip++) {
      std::vector<double> ParForward = parameters;
      std::vector<double> ParBackward = parameters;

      ParForward[ip] = parameters[ip] * ( 1 + eps );
      ParBackward[ip] = parameters[ip] * ( 1 - eps );

      std::vector<double> fp = f(x, ParForward);
      std::vector<double> fm = f(x, ParBackward);
      std::transform(fp.begin(), fp.end(), fm.begin(), results.begin() + Counter, [&] (double fplus, double fminus)
      {
        return 0.5 * (fplus - fminus) / eps / parameters[ip];
      });

      Counter += fp.size();
    }
    return results;
  }
}