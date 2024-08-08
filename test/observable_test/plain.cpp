#include "NTK/Observable.h"
#include "NNAD/FeedForwardNN.h"
#include "NTK/utility_test.hpp"

int main() {
  const std::vector<int> arch{1, 20, 20, 1};

  // Initialise network
  std::unique_ptr<NTK::NNAD> nn = std::make_unique<NTK::NNAD>(arch, 0, nnad::OutputFunction::LINEAR, false);

  int batch_size = 4;
  std::vector<NTK::data> data_batch(4);
  data_batch[0] = {0.1};
  data_batch[1] = {0.17};
  data_batch[2] = {0.3};
  data_batch[3] = {0.5};
  std::cout << "Sono qui" << std::endl;

  NTK::dNN dnn(nn.get(), data_batch);
  NTK::ddNN ddnn(nn.get(), data_batch);
  NTK::d3NN d3nn(nn.get(), data_batch);
  NTK::O2 o2(batch_size, arch.back());
  NTK::O3 o3(batch_size, arch.back());
  NTK::O4 o4(batch_size, arch.back());
  dnn.Evaluate();
  ddnn.Evaluate();
  d3nn.Evaluate();
  //o2.Evaluate(&dnn);
  //o3.Evaluate(&dnn, &ddnn);
  o4.Evaluate(&dnn, &ddnn, &d3nn);

  return 0;
}