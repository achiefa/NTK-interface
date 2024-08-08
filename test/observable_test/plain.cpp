#include "NTK/Observable.h"
#include "NNAD/FeedForwardNN.h"
#include "NTK/utility_test.hpp"

int main() {
  const std::vector<int> arch{1, 2, 1};

  // Initialise network
  std::unique_ptr<NTK::NNAD> nn = std::make_unique<NTK::NNAD>(arch, 0, nnad::OutputFunction::LINEAR, false);
  int nout = arch.back();
  int np = nn->GetParameterNumber();
  double b1 = nn->GetBias(1,0);
  double b2 = nn->GetBias(1,1);
  double B  = nn->GetBias(2,0);
  double w1 = nn->GetLink(1, 0, 0);
  double w2 = nn->GetLink(1, 1, 0);
  double W1 = nn->GetLink(2, 0, 0);
  double W2 = nn->GetLink(2, 0, 1);
  std::cout << "b1 " << b1 << std::endl;
  std::cout << "b2 " << b2 << std::endl;
  std::cout << "w1 " << w1 << std::endl;
  std::cout << "w2 " << w2 << std::endl;
  std::cout << "B " << B << std::endl;
  std::cout << "W1 " << W1 << std::endl;
  std::cout << "W1 " << W2 << std::endl;

  std::map<std::string, int> map = nn->GetStrIntMap();

  std::vector<double> pars = nn->GetParameters();
  std::cout << "pars" << std::endl;
  for (int ip=0; ip < pars.size(); ip++){
    std::cout << pars[ip] << std::endl;
  }

  for (auto& x: map) {
    std::cout << x.first << ": " << x.second << '\n';
  }
  return 0;
}