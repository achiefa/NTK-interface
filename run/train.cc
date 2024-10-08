#include "NTK/AnalyticCostFunction.h"
#include "NTK/IterationCallBack.h"
#include "NTK/Timer.h"
#include "NTK/direxists.h"

#include <getopt.h>
#include <sys/stat.h>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include "NTK/NumericalDerivative.h"
#include "NTK/Observable.h"
#include "NTK/utility.hpp"


double P10(double x) { return (1. / 256) * (46189 * pow(x, 10) - 109395 * pow(x, 8) + 90090 * pow(x, 6) - 30030 * pow(x, 4) + 3465 * pow(x, 2) - 63); }

int main(int argc, char *argv[])
{
  const char* const short_opts = "i";
  const option long_opts[] =
  {
    {"initialisation", no_argument, nullptr, 'i'},
  };

  bool only_initialisation = false;

  while (true)
    {
      const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

      if (opt == -1)
        break;

      switch (opt)
        {
          case 'i':
            only_initialisation = true;
            break;
          case '?':
          default:
            std::cerr << "Usage of " << argv[0] << ": [-i|--initialisation] <replica index> <fit folder>" << std::endl;
            exit(-1);
        }
    }

  if ((argc - optind) != 2)
    {
      std::cerr << "Usage of " << argv[0] << ": [-i|--initialisation] <replica index> <fit folder>" << std::endl;
      exit(-1);
    }

  // Input information
  int replica = atoi(argv[optind]);
  const std::string FitFolder = argv[optind + 1];

  // Require that the result folder exists. If not throw an exception. 
  namespace fs = std::filesystem; 
  if (!NTK::is_dir(FitFolder))
    {
      std::cerr << "Folder " << FitFolder.c_str() << " does not exist.\n";
      exit(-1);
    }

  const std::string InputCardPath = FitFolder + "/meta.yaml";
  const std::string DataFile = FitFolder + "/Data.yaml";

  // Read Input Card
  YAML::Node InputCard = YAML::LoadFile(InputCardPath);

  // ====================================================
  //          Load the dataset
  // ====================================================
  YAML::Node InputData = YAML::LoadFile(DataFile);
  int ndata = InputData["Metadata"]["Number of points"].as<int>();
  NTK::vecdata Data;
  for (int i = 0; i < ndata; i++)
  { 
    NTK::Datapoint tuple;
    std::get<0>(tuple) = InputData["Independent variables"][i]["Value"].as<NTK::dvec>();
    std::get<1>(tuple) = InputData["Dependent variables"][i]["Value"].as<NTK::dvec>();
    std::get<2>(tuple) = InputData["Dependent variables"][i]["Noise"].as<NTK::dvec>();

    Data.push_back(tuple);
  }

  // ====================================================
  //          Neural Network initialisation
  // ====================================================
  int Seed = InputCard["Seed"].as<int>() + replica;
  std::vector<int> NNarchitecture = InputCard["NNarchitecture"].as<std::vector<int>>();
  nnad::FeedForwardNN<double> *nn = new nnad::FeedForwardNN<double> (NNarchitecture, Seed, false,
                                        nnad::Tanh<double>,  nnad::dTanh<double>,
                                        nnad::OutputFunction::LINEAR, nnad::InitDistribution::GAUSSIAN, {}, true);

  // ============================================================
  // Run the solver with some options.
  // ============================================================

  // Set parameters for Ceres Solver
    const int np = nn->GetParameterNumber();
    const std::vector<double> pars = nn->GetParameters();
    std::vector<double *> initPars(np);
    for (int ip = 0; ip < np; ip++)
        initPars[ip] = new double(pars[ip]);

  // Allocate a "Chi2CostFunction" instance to be fed to ceres for minimisation
  NTK::AnalyticCostFunction *analytic_chi2cf = new NTK::AnalyticCostFunction{nn, Data};

  // Compute initial chi24
  std::cout << "Initial chi2 = " << analytic_chi2cf->Evaluate() << std::endl;
  std::cout << "\n";

  //======================
  // Observables
  //======================
  int batch_size = 4;
  std::vector<NTK::data> data_batch = NTK::create_data_batch(batch_size, Data);
  NTK::dNN dnn(nn, data_batch);
  NTK::ddNN ddnn(nn, data_batch);
  NTK::d3NN d3nn(nn, data_batch);
  NTK::O2 o2(batch_size, NNarchitecture.back());
  NTK::O3 o3(batch_size, NNarchitecture.back());
  NTK::O4 o4(batch_size, NNarchitecture.back());
  dnn.Evaluate();
  ddnn.Evaluate();
  d3nn.Evaluate();
  o2.Evaluate(&dnn);
  o3.Evaluate(&dnn, &ddnn);
  o4.Evaluate(&dnn, &ddnn, &d3nn);
  print_obs_to_yaml(FitFolder, 0, replica, &o2, &o3, &o4);
  


  // Solve
  if (!only_initialisation){
    // Allocate "Problem" instance
    ceres::Problem problem;
    problem.AddResidualBlock(analytic_chi2cf, NULL, initPars);

    ceres::Solver::Options options;
    options.minimizer_type = ceres::LINE_SEARCH;
    options.line_search_direction_type = ceres::STEEPEST_DESCENT;
    options.line_search_type = ceres::ARMIJO;
    options.max_num_iterations = InputCard["max_num_iterations"].as<int>();
    options.minimizer_progress_to_stdout = true;
    options.function_tolerance  = 1e-7;
    options.gradient_tolerance  = 1e-7;
    options.parameter_tolerance = 1e-7;
    //options.num_threads = 4;

    // Iteration callback
    options.update_state_every_iteration = true;
    NTK::IterationCallBack *callback = new NTK::IterationCallBack(false, FitFolder, replica, initPars, analytic_chi2cf);
    options.callbacks.push_back(callback);

    ceres::Solver::Summary summary;

    Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << "\n";

    // Compute final chi2
    std::vector<double> final_pars;
    for (int i = 0; i < np; i++)
      final_pars.push_back(initPars[i][0]);
    nn->SetParameters(final_pars);
    std::cout << "Final chi2 = " << analytic_chi2cf->Evaluate() << std::endl;
    std::cout << "\n";

    YAML::Emitter emitter;
      emitter << YAML::BeginMap;
      //__________________________________
      emitter << YAML::Key << "Dependent variables";
      emitter << YAML::Value << YAML::BeginSeq;
      for (auto &d : Data)
      {
          emitter << YAML::BeginMap;
          emitter << YAML::Key << "Value" << YAML::Value << nn->Evaluate(std::get<0>(d));
          emitter << YAML::EndMap;
      }
      emitter << YAML::EndSeq;
      emitter << YAML::Newline << YAML::Newline;
      emitter << YAML::Key << "Independent variables";
      emitter << YAML::Value << YAML::BeginSeq;
      for (auto &d : Data)
      {
          emitter << YAML::BeginMap;
          emitter << YAML::Key << "Value" << YAML::Value << std::get<0>(d);
          emitter << YAML::EndMap;
      }
      //__________________________________
      emitter << YAML::EndMap;

      std::ofstream fout;
      fout = std::ofstream(FitFolder + "/output/Output_" + std::to_string(replica) + ".yaml", std::ios::out | std::ios::app);
      fout << emitter.c_str();
      fout.close();
      delete callback;
  }

  delete nn;
  delete analytic_chi2cf;
  return 0;
}