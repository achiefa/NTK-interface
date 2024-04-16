#include "NTK/AnalyticCostFunction.h"
#include "NTK/IterationCallBack.h"
#include "NTK/Timer.h"
#include "NTK/direxists.h"

#include <getopt.h>
#include <sys/stat.h>
#include <iostream>


double P10(double x) { return (1. / 256) * (46189 * pow(x, 10) - 109395 * pow(x, 8) + 90090 * pow(x, 6) - 30030 * pow(x, 4) + 3465 * pow(x, 2) - 63); }
int main(int argc, char *argv[])
{
  // SET A FLAG TO STOP AT INITIALISATION
  if ((argc - optind) != 4)
    {
      std::cerr << "Usage of " << argv[0] << ": <replica index> <runcard> <result folder> <data file>" << std::endl;
      exit(-1);
    }

  // Input information
  int replica = atoi(argv[optind]);
  const std::string InputCardPath = argv[optind + 1];
  const std::string ResultFolder = argv[optind + 2];
  const std::string DataFile = argv[optind + 3];

  // Assign to the fit the name of the input card
  const std::string OutputFolder = ResultFolder + "/" + InputCardPath.substr(InputCardPath.find_last_of("/") + 1, InputCardPath.find(".yaml") - InputCardPath.find_last_of("/") - 1);

  // Require that the result folder exists. If not throw an exception. 
  namespace fs = std::filesystem; 
  if (NTK::is_dir(ResultFolder))
  {
    printf("Folder %s does not exist. Creating a new one... \n", ResultFolder.c_str());
    fs::create_directories(ResultFolder);
  }

  // Read Input Card
  YAML::Node InputCard = YAML::LoadFile(InputCardPath);

  // ====================================================
  //          Neural Network initialisation
  // ====================================================
  int Seed = InputCard["Seed"].as<int>() + replica;
  std::vector<int> NNarchitecture = InputCard["NNarchitecture"].as<std::vector<int>>();
  nnad::FeedForwardNN<double> *nn = new nnad::FeedForwardNN<double> (NNarchitecture, Seed, true,
                                        nnad::Sigmoid<double>,  nnad::dSigmoid<double>, 
                                        nnad::OutputFunction::LINEAR, nnad::InitDistribution::GAUSSIAN, {}, true);

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

  // ============================================================
  // Run the solver with some options.
  // ============================================================

  // // Set parameters for Ceres Solver
  //   const int np = nn->GetParameterNumber();
  //   const std::vector<double> pars = nn->GetParameters();
  //   setd::vector<double *> initPars(np);
  //   for (int ip = 0; ip < np; ip++)
  //       initPars[ip] = new double(pars[ip]);

  // // Allocate a "Chi2CostFunction" instance to be fed to ceres for minimisation
  // ceres::CostFunction *analytic_chi2cf = nullptr;
  // analytic_chi2cf = new NTK::AnalyticCostFunction(np, Data, NNarchitecture, Seed);
  // problem.AddResidualBlock(analytic_chi2cf, NULL, initPars);


  //   // Allocate "Problem" instance
  //   ceres::Problem problem;

  // // Compute initial chi2
  // double chi2 = 0;
  // std::vector<std::vector<double>> Predictions;
  // for (int i = 0; i < n; i++)
  // {
  //   std::vector<double> x;
  //   x.push_back(std::get<0>(Data[i]));
  //   std::vector<double> v = nn->Evaluate(x);
  //   Predictions.push_back(v);
  // }
  // // NOT SURE
  // for (int id = 0; id < n; id++)
  //   chi2 += pow((Predictions[id][0] - std::get<1>(Data[id])) / std::get<2>(Data[id]), 2);
  // chi2 /= n;
  // std::cout << "Initial chi2 = " << chi2 << endl;
  // std::cout << "\n";

  // // Solve
  // ceres::Solver::Options options;
  // options.max_num_iterations = InputCard["max_num_iterations"].as<int>();
  // options.minimizer_progress_to_stdout = true;
  // options.function_tolerance = InputCard["function_tolerance"].as<double>();
  // options.parameter_tolerance = 1e-20;
  // options.gradient_tolerance = 1e-20;
  // ceres::Solver::Summary summary;
  // // Timer
  // Timer t;
  // Solve(options, &problem, &summary);
  // double duration = t.stop();
  // cout << summary.FullReport() << "\n";

  // // Compute final chi2
  // chi2 = 0;
  // std::vector<double> final_pars;
  // for (int i = 0; i < np; i++)
  //   final_pars.push_back(initPars[i][0]);
  // nn->SetParameters(final_pars);

  // for (int i = 0; i < n; i++)
  // {
  //   std::vector<double> x;
  //   x.push_back(std::get<0>(Data[i]));
  //   std::vector<double>
  //       v = nn->Evaluate(x);
  //   Predictions.at(i) = v;
  // }
  // ofstream test("test.dat");
  // for (int id = 0; id < n; id++)
  // {
  //   chi2 += pow((Predictions[id][0] - std::get<1>(Data[id])) / std::get<2>(Data[id]), 2);
  //   test << std::get<0>(Data[id]) << " " << Predictions[id][0] << " " << std::get<1>(Data[id]) << " " << truth[id] << " " << std::get<2>(Data[id]) << std::endl;
  // }

  // chi2 /= n;
  // std::cout << "Final chi2 = " << chi2 << std::endl;
  // std::cout << "Number of parameters = "<< np << std::endl;
  // std::cout << "\n";


  delete nn;
  return 0;
}