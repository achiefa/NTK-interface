#include "NTK/AnalyticCostFunction.h"
#include "NTK/Observable.h"
#include <utility>

namespace NTK
{
  std::vector<data> create_data_batch(const int& batch_size, vecdata Data_vector) {
    std::vector<data> res;
    int StepSize = int(Data_vector.size() / batch_size);
    for(size_t a=0; a < batch_size; a++)
      res.push_back(std::get<0>(Data_vector[(a) * StepSize]));

    return res;
  }


  template<typename ... Args>
  void print_obs_to_yaml(const std::string &FitFolder, const int &iteration, const int &replica, Args&& ... args) {
    // Output parameters into yaml file

    std::vector< std::pair< std::string, std::vector<double>> > name_tensor_pairs;
    ([&]
    {
        auto tensor = args->GetTensor();
        std::vector<double> tensor_vec ( tensor.data(),  tensor.data() + tensor.size() );
        std::pair<std::string, std::vector<double>> tensor_pair;
        tensor_pair.first = args->GetID();
        std::cout << tensor_pair.first << std::endl;
        tensor_pair.second = tensor_vec;
        name_tensor_pairs.push_back(tensor_pair);
    } (), ...);

    YAML::Emitter emitter;
    emitter << YAML::BeginSeq;
    emitter << YAML::Flow << YAML::BeginMap;
    emitter << YAML::Key << "iteration" << YAML::Value << int(iteration);
    for (auto& pr : name_tensor_pairs) {
      emitter << YAML::Key << pr.first << YAML::Value << YAML::Flow << pr.second;
    }
    emitter << YAML::EndMap;
    emitter << YAML::EndSeq;
    emitter << YAML::Newline;
    std::ofstream fout(FitFolder + "/log/replica_" + std::to_string(replica) + ".yaml", std::ios::out | std::ios::app);
    fout << emitter.c_str();
    fout.close();
    return;
  }
}