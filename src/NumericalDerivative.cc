#include "NTK/NumericalDerivative.h"

namespace NTK
{
  //__________________________________________________________________________________________
  std::vector<std::vector<double>> FiniteDifference (
  std::function< std::vector<double> (std::vector<double> const&, std::vector<double>)> f,
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
  std::vector<double> FiniteSecondDifferenceVec (
    std::function<std::vector<double> (std::vector<double> const&, std::vector<double>)> f,
    std::vector<double> parameters,
    std::vector<double> const& x,
    int const& size_f,
    double const& eps)
  {
    int np = int(parameters.size());
    std::vector<double> results (np * np * size_f, 0.);
    int counter = 0;

    for (int ip = 0; ip < np; ip++) {
      for (int jp = 0; jp < np; jp++) {
        std::vector<double> ParFF = parameters;
        std::vector<double> ParFB = parameters;
        std::vector<double> ParBF = parameters;
        std::vector<double> ParBB = parameters;
        ParFF[ip] +=  parameters[ip] * eps;
        ParFF[jp] +=  parameters[jp] * eps;
        ParFB[ip] +=  parameters[ip] * eps;
        ParFB[jp] -=  parameters[jp] * eps;
        ParBF[ip] -=  parameters[ip] * eps;
        ParBF[jp] +=  parameters[jp] * eps;
        ParBB[ip] -=  parameters[ip] * eps;
        ParBB[jp] -=  parameters[jp] * eps;
        std::vector<double> f_FF = f(x, ParFF);
        std::vector<double> f_FB = f(x, ParFB);
        std::vector<double> f_BF = f(x, ParBF);
        std::vector<double> f_BB = f(x, ParBB);
        std::vector<double> f_FF_BB (f_FF.size(), 0.);
        std::vector<double> f_FB_BF (f_FF.size(), 0.);
        std::vector<double> res (f_FF.size(), 0.);
        std::transform(f_FF.begin(), f_FF.end(), f_BB.begin(), f_FF_BB.begin(), [&] (double f1, double f2) {return f1 + f2;});
        std::transform(f_FB.begin(), f_FB.end(), f_BF.begin(), f_FB_BF.begin(), [&] (double f1, double f2) {return f1 + f2;});
        std::transform(f_FF_BB.begin(), f_FF_BB.end(), f_FB_BF.begin(), results.begin() + counter, [&] (double f1, double f2)
        { 
          return (f1 - f2) / 4. / eps / eps / parameters[ip] / parameters[jp];
        });
        counter += f_FF.size();
      }
    }
    return results;
  }

  //__________________________________________________________________________________________
  std::vector<double> helper::HelperSecondFiniteDer (nnad::FeedForwardNN<double> *NN,
                                          std::vector<double> input,
                                          double const& eps)
    {
    // Initialise std::function for second derivative
    std::function<std::vector<double>(std::vector<double> const&, std::vector<double>)> dNN_func
    {
      [&] (std::vector<double> const& x, std::vector<double> parameters) -> std::vector<double>
      {
        nnad::FeedForwardNN<double> aux_nn{*NN}; // Requires pointer dereference
        aux_nn.SetParameters(parameters);
        return aux_nn.Derive(x);
      }
    };
    int np = NN->GetParameterNumber();
    int nout = NN->GetArchitecture().back();
    return FiniteDifferenceVec(dNN_func, NN->GetParameters(), input, np * nout + nout, eps);
    }


    // NEW STUFF
    //________________________________
    typedef std::function<std::vector<double>(std::vector<double> const&, std::vector<double>)> Functor;
    std::vector<double> dNNAD_cleaner(nnad::FeedForwardNN<double> *NN, int const& nout, std::vector<double> const& x, std::vector<double> parameters)
    {
      nnad::FeedForwardNN<double> aux_nn{*NN}; // Requires pointer dereference
      aux_nn.SetParameters(parameters);
      std::vector<double> res = aux_nn.Derive(x);
      res.erase(res.begin(), res.begin() + nout);
      return res; 
    }
    

    std::vector<double> helper::HelperSecondFiniteDer2 (nnad::FeedForwardNN<double> *NN,
                                          std::vector<double> input,
                                          double const& eps)
    { 
      int nout = NN->GetArchitecture().back();
      int np = NN->GetParameterNumber();

      // Define functor
      std::function<std::vector<double>(std::vector<double> const&, std::vector<double>)> dNNAD_wrapper
      {
        [&] (std::vector<double> const& x, std::vector<double> parameters) -> std::vector<double>
        {
          return dNNAD_cleaner(NN, nout, x, parameters);
        }
      };


      return FiniteDifferenceVec(dNNAD_wrapper, NN->GetParameters(), input, np * nout, eps);
    }

    //__________________________________________________________________________________________
    std::vector<double> helper::HelperThirdFiniteDer (nnad::FeedForwardNN<double> *NN,
                                          std::vector<double> input,
                                          double const& eps)
    {
      int nout = NN->GetArchitecture().back();
      int np = NN->GetParameterNumber();

      // Define functor
      std::function<std::vector<double>(std::vector<double> const&, std::vector<double>)> dNNAD_wrapper
      {
        [&] (std::vector<double> const& x, std::vector<double> parameters) -> std::vector<double>
        {
          return dNNAD_cleaner(NN, nout, x, parameters);
        }
      };

      return FiniteSecondDifferenceVec(dNNAD_wrapper, NN->GetParameters(), input, np * nout, eps);
    }
}