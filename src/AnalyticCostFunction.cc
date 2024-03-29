#include "NTK/AnalyticCostFunction.h"

namespace NTK
{
    //____________________________________________
    AnalyticCostFunction::AnalyticCostFunction(nnad::FeedForwardNN<double> *NN,
                        vecdata const &Data,
                        bool NUMERIC) : _nn(NN),
                                        _Data(Data)
                                        
    {
        _Np = _nn->GetParameterNumber();
        _ndata = _Data.size();
        _OutSize = _nn->GetArchitecture().back();

        if (!NUMERIC)
            {
                set_num_residuals(int(_ndata));

                for (int ip = 0; ip < _Np; ip++)
                    mutable_parameter_block_sizes()->push_back(1);
            }
    }

  //_________________________________________________________________________________
  double AnalyticCostFunction::GetResidual(const int& id) const
  {
    const std::vector<double> input = std::get<0>(_Data[id]);
    const std::vector<double> target = std::get<1>(_Data[id]);
    const std::vector<double> error = std::get<2>(_Data[id]);
    const std::vector<double> nnx = _nn->Evaluate(input);

    std::vector<double> res;
    std::transform(nnx.begin(), nnx.end(), target.begin(), res.begin(), [] (double const& n, double const& t) { return n - t; });
    std::transform(res.begin(), res.end(), error.begin(), res.begin(), [] (double const& r, double const& e) { return r / e; });

    return sqrt(std::inner_product(res.begin(), res.end(), res.begin(), double(0.0)));;
  }

  //_________________________________________________________________________________
  std::vector<double> AnalyticCostFunction::GetResiduals() const
  {
    std::vector<double> res (_ndata, 0.);
    for (int id = 0; id < _ndata; id++)
      res[id] = GetResidual(id);
    
    return res;
  }

    //_________________________________________________________________________
  bool AnalyticCostFunction::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
  {
    std::vector<double> pars(_Np);

    for (int i = 0; i < _Np; i++)
      pars[i] = parameters[i][0];

    _nn->SetParameters(pars);

    // Residuals and Jacobian
    if (jacobians != NULL)
    { 
        // For each point in the data set
        for (int id = 0; id < _ndata; id++)
            {

            const std::vector<double> input = std::get<0>(_Data[id]);
            //const std::vector<double> target = std::get<1>(_Data[id]);
            const std::vector<double> error = std::get<2>(_Data[id]);

            //const std::vector<double> nnx = _nn->Evaluate(input);
            
            // Generalise for multiple output layers
            // Each point in the dataset may be R^n. In order to
            // compute the residual for a n-dimensional vector, 
            // I will compute norm of the difference, weighted
            // by the std.
            //std::vector<double> res;
            //std::transform(nnx.begin(), nnx.end(), target.begin(), res.begin(), [] (double const& n, double const& t) { return n - t; });
            //std::transform(res.begin(), res.end(), error.begin(), res.begin(), [] (double const& r, double const& e) { return r / e; });
            residuals[id] = GetResidual(id); //sqrt(std::inner_product(res.begin(), res.end(), res.begin(), double(0.0)));
            const std::vector<double> dnnx = _nn->Derive(input);

            for (int ip = 0; ip < _Np; ip++){
              jacobians[ip][id] = double(0);
              for (int i = 0; i < _OutSize; i++){
                // TODO Check this funciton
                jacobians[ip][id] += dnnx[i + _OutSize * (ip + 1)] / error[i];
              }
            }
          }
    }

    // Only residuals
    else
    {
        for (int id = 0; id < _ndata; id++)
            {
              //const std::vector<double> input = std::get<0>(_Data[id]);
              //const std::vector<double> target = std::get<1>(_Data[id]);
              //const std::vector<double> error = std::get<2>(_Data[id]);
              //const std::vector<double> nnx = _nn->Evaluate(input);

              //std::vector<double> res;
              //std::transform(nnx.begin(), nnx.end(), target.begin(), res.begin(), [] (double const& n, double const& t) { return n - t; });
              //std::transform(res.begin(), res.end(), error.begin(), res.begin(), [] (double const& r, double const& e) { return r / e; });
              residuals[id] = GetResidual(id);//sqrt(std::inner_product(res.begin(), res.end(), res.begin(), double(0.0)));
            }
    }
    return true;
  }

  //_________________________________________________________________________________
  double AnalyticCostFunction::Evaluate() const
  {
    // Initialise chi2 and number of data points

    const std::vector<double> res = GetResiduals();
    double chi2 = std::inner_product(res.begin(), res.end(), res.begin(), 0.);

    return chi2 / _ndata;
  }

  //_________________________________________________________________________
  bool AnalyticCostFunction::operator()(double const *const *parameters, double *residuals) const
  {
    std::vector<double> pars(_Np);

    for (int i = 0; i < _Np; i++)
      pars[i] = parameters[i][0];

    _nn->SetParameters(pars); // Set parameters of the NN

    for (int id = 0; id < _ndata; id++)
      residuals[id] = GetResidual(id);

    return true;
  }
}
