//
// Author: Rabah Abdul Khalek - rabah.khalek@gmail.com
//

#include "NTK/NumericCostFunction.h"

NumericCostFunction::~NumericCostFunction(){delete _nn;}

NumericCostFunction::NumericCostFunction(int const &Np,
                                         vecdata const &Data,
                                         std::vector<int> const &NNarchitecture,
                                         int const &Seed) : _Np(Np),
                                                            _Data(Data),
                                                            _NNarchitecture(NNarchitecture),
                                                            _Seed(Seed)
{
    _nn = new nnad::FeedForwardNN<double>(_NNarchitecture, _Seed, false);
    _ndata = _Data.size();
    _OutSize = _NNarchitecture.back();
}

bool NumericCostFunction::operator()(double const *const *parameters, double *residuals) const
{
    std::vector<double> pars(_Np);
    for (int i = 0; i < _Np; i++)
        pars[i] = parameters[i][0];

    _nn->SetParameters(pars);

    for (int id = 0; id < (int) _Data.size(); id++)
    {
        residuals[id] = GetResidual(id);
    }

    return true;
}


//_________________________________________________________________________________
  double NumericCostFunction::GetResidual(const int& id) const
  {
    const std::vector<double> input = std::get<0>(_Data[id]);
    const std::vector<double> target = std::get<1>(_Data[id]);
    const std::vector<double> error = std::get<2>(_Data[id]);
    const std::vector<double> nnx = _nn->Evaluate(input);
    //for (int i=0; i<nnx.size(); i++){
    //  std::cout << "_________________" << std::endl;
    //  std::cout << "Target : " << target[i] << std::endl;
    //  std::cout << "Theory : " << nnx[i] << std::endl;
    //  std::cout << "Error : " << error[i] << std::endl;
    //}

    std::vector<double> res (target.size(), 0.);
    std::transform(nnx.begin(), nnx.end(), target.begin(), res.begin(), [] (double const& n, double const& t) { return n - t; });
    std::transform(res.begin(), res.end(), error.begin(), res.begin(), [] (double const& r, double const& e) { return r / e; });
    double result = sqrt(std::inner_product(res.begin(), res.end(), res.begin(), double(0.0)));
    //printf(KRED "Residue " KNRM " : %f \n", result);
    return result;
  }

  //_________________________________________________________________________________
  std::vector<double> NumericCostFunction::GetResiduals() const
  {
    std::vector<double> res (_ndata, 0.);
    for (int id = 0; id < _ndata; id++)
      res[id] = GetResidual(id);
  
    return res;
  }