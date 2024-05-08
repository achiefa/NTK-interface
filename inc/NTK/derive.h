// NNAD

#include <fstream>
#include <functional>
#include <algorithm>

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
      for (int jp = 0; jp < np; jp++)
        {
          ParForward[jp] = (jp == ip ? parameters[jp] * ( 1 + eps ) : parameters[jp]);
          ParBackward[jp] = (jp == ip ? parameters[jp] * ( 1 - eps ) : parameters[jp]);
        }

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
}