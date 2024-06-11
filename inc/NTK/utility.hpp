#include "NTK/AnalyticCostFunction.h"
#include "NTK/Observable.h"

namespace NTK
{
  std::vector<data> create_data_batch(const int& batch_size, vecdata Data_vector) {
    std::vector<data> res;
    int StepSize = int(Data_vector.size() / batch_size);
    for(size_t a=0; a < batch_size; a++)
      res.push_back(std::get<0>(Data_vector[(a) * StepSize]));

    return res;
  }


  void print_to_yaml(std::string FitFolder, O2* ntk, O3* o3) {
    // Output parameters into yaml file

    auto ntk_tensor = ntk->GetTensor();
    auto O3_tensor = ntk->GetTensor();
    std::vector<double> NTK_vec ( ntk_tensor.data(),  ntk_tensor.data() + ntk_tensor.size() );
    std::vector<double> O3_vec ( O3_tensor.data(),  O3_tensor.data() + O3_tensor.size() );

    YAML::Emitter emitter;
    emitter << YAML::BeginSeq;
    emitter << YAML::Flow << YAML::BeginMap;
    emitter << YAML::Key << "iteration" << YAML::Value << int(0);
    emitter << YAML::Key << "NTK" << YAML::Value << YAML::Flow << NTK_vec;
    emitter << YAML::Key << "O_3" << YAML::Value << YAML::Flow << O3_vec;
    emitter << YAML::EndMap;
    emitter << YAML::EndSeq;
    emitter << YAML::Newline;
    std::ofstream fout(FitFolder + "/log/replica_" + std::to_string(0) + ".yaml", std::ios::out | std::ios::app);
    fout << emitter.c_str();
    fout.close();
    return;
  }
}