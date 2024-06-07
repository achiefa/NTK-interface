

#include "NTK/Observable.h"
#include "NTK/NumericalDerivative.h"
#include "NNAD/FeedForwardNN.h"




int main() {
  const std::vector<int> arch{3, 2, 10};
  std::unique_ptr<NTK::NNAD> nn = std::make_unique<NTK::NNAD>(arch, 0, nnad::OutputFunction::ACTIVATION, true);

  int batch_size = 4;
  int nout = arch.back();
  int np = nn->GetParameterNumber();
  std::vector<NTK::data> data_vector (batch_size);
  data_vector[0] = {0.0, 0.3, 0.6};
  data_vector[1] = {0.0, 0.7, 0.9};
  data_vector[2] = {0.8, 0.1, 0.3};
  data_vector[3] = {0.1, 0.9, 0.5};

  NTK::ddNN ddnn(batch_size, nout, np);
  NTK::ddNN ddnn2(batch_size, nout, np);

  ddnn.Evaluate(data_vector, nn.get());

  ddnn2.Evaluate(data_vector, nn.get());

  std::cout << "_________________" << std::endl;


  //delete nn;
  return 0;
}