#include "NTK/NumericalDerivative.h"

namespace NTK
{
  //__________________________________________________________________________________________
  std::vector<std::vector<double>> FiniteDifference (
  std::function<std::vector<double> (std::vector<double> const&, std::vector<double>)> f,
  std::vector<double> parameters, 
  std::vector<double> const& x, 
  double const& eps)
  {
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

//__________________________________________________________________________________________
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

  //__________________________________________________________________________________________
  std::vector<double> helper::HelperSecondFiniteDer (nnad::FeedForwardNN<double> *NN,
                                          std::vector<double>parameters,
                                          std::vector<double> input,
                                          int const& Np,
                                          int const& Nout,
                                          double const& eps)
    {
    // Initialise std::function for second derivative
    std::function<std::vector<double>(std::vector<double> const&, std::vector<double>)> dNN
    {
      [&] (std::vector<double> const& x, std::vector<double> parameters) -> std::vector<double>
      {
        nnad::FeedForwardNN<double> aux_nn{*NN}; // Requires pointer dereference
        aux_nn.SetParameters(parameters);
        return aux_nn.Derive(x);
      }
    };
    return FiniteDifferenceVec(dNN, parameters, input, Np + Nout, eps);
    }
}