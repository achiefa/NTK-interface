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
  if ((argc - optind) != 3)
    {
      std::cerr << "Usage of " << argv[0] << ": <replica index> <runcard> <result folder>" << std::endl;
      exit(-1);
    }

  // Input information
  int replica = atoi(argv[optind]);
  const std::string InputCardPath = argv[optind + 1];
  //const std::string DataFolder = argv[optind + 2];
  const std::string ResultFolder = argv[optind + 2];

  // Assign to the fit the name of the input card
  const std::string OutputFolder = ResultFolder + "/" + InputCardPath.substr(InputCardPath.find_last_of("/") + 1, InputCardPath.find(".yaml") - InputCardPath.find_last_of("/") - 1);

  // Require that the result folder exists. If not throw an exception.  
  if (NTK::is_dir(ResultFolder))
    printf("Folder %s does not exist. Creating a new one... \n", ResultFolder.c_str());

  // Read Input Card
  YAML::Node config = YAML::LoadFile(InputCardPath);

  // Set the seeds based on replica wanted
  //config["NNAD"]["seed"] = config["NNAD"]["seed"].as<int>() + replica;

  // ====================================================
  //          Neural Network initialisation
  // ====================================================
  int Seed = InputCard["Seed"].as<int>();
  std::vector<int> NNarchitecture = InputCard["NNarchitecture"].as<vector<int>>();
  nnad::FeedForwardNN<double> *nn = new nnad::FeedForwardNN<double> (NNarchitecture, Seed, true,
                                        nnad::Sigmoid<double>,  nnad::dSigmoid<double>, 
                                        nnad::OutputFunction::LINEAR, nnad::InitDistribution::GAUSSIAN, {}, true);

  // ====================================================
  //          Data generation and fluctuations
  // ====================================================
  int n = // number of points
  NTK::vecdata Data;
  double xmin = InputCard["xmin"].as<double>();
  double xmax = InputCard["xmax"].as<double>();
  double yshift = InputCard["yshift"].as<double>();

  // noise
  double noise_mean = InputCard["noise_mean"].as<double>();
  double noise_sd = InputCard["noise_sd"].as<double>();
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::normal_distribution<double> noise_gen(noise_mean, noise_sd);

  std::vector<double> truth;
  for (int i = 0; i < n; i++)
  {
    NTK::Datapoint tuple;
    double x = xmin + i * (xmax - xmin) / n;
    double y = P10(x) + yshift;
    truth.push_back(y);

    // noise
    double noise = 0;
    while(! noise)
      if (y != 0)
        noise = noise_gen(rng)*y;
      else
        noise = noise_gen(gen);
    
    get<0>(tuple) = x;
    get<1>(tuple) = y+noise;
    get<2>(tuple) = std::abs(noise);
    Data.push_back(tuple);
  }

  // Set parameters for Ceres Solver
  const int np = nn->GetParameterNumber();
  const std::vector<double> pars = nn->GetParameters();
  setd::vector<double *> initPars(np);
  for (int ip = 0; ip < np; ip++)
    initPars[ip] = new double(pars[ip]);

  // Allocate "Problem" instance
  ceres::Problem problem;

  // Allocate a "Chi2CostFunction" instance to be fed to ceres for minimisation
  ceres::CostFunction *analytic_chi2cf = nullptr;
  analytic_chi2cf = new AnalyticCostFunction(np, Data, NNarchitecture, Seed);
  problem.AddResidualBlock(analytic_chi2cf, NULL, initPars);

  // ============================================================
  // Run the solver with some options.
  // ============================================================
  // Compute initial chi2
  double chi2 = 0;
  std::vector<std::vector<double>> Predictions;
  for (int i = 0; i < n; i++)
  {
    std::vector<double> x;
    x.push_back(get<0>(Data[i]));
    std::vector<double> v = nn->Evaluate(x);
    Predictions.push_back(v);
  }
  // NOT SURE
  for (int id = 0; id < n; id++)
    chi2 += pow((Predictions[id][0] - get<1>(Data[id])) / get<2>(Data[id]), 2);
  chi2 /= n;
  std::cout << "Initial chi2 = " << chi2 << endl;
  std::cout << "\n";

  // Solve
  ceres::Solver::Options options;
  options.max_num_iterations = InputCard["max_num_iterations"].as<int>();
  options.minimizer_progress_to_stdout = true;
  options.function_tolerance = InputCard["function_tolerance"].as<double>();
  options.parameter_tolerance = 1e-20;
  options.gradient_tolerance = 1e-20;
  ceres::Solver::Summary summary;
  // Timer
  Timer t;
  Solve(options, &problem, &summary);
  double duration = t.stop();
  cout << summary.FullReport() << "\n";

  // Compute final chi2
  chi2 = 0;
  std::vector<double> final_pars;
  for (int i = 0; i < np; i++)
    final_pars.push_back(initPars[i][0]);
  nn->SetParameters(final_pars);

  for (int i = 0; i < n; i++)
  {
    std::vector<double> x;
    x.push_back(get<0>(Data[i]));
    std::vector<double>
        v = nn->Evaluate(x);
    Predictions.at(i) = v;
  }
  ofstream test("test.dat");
  for (int id = 0; id < n; id++)
  {
    chi2 += pow((Predictions[id][0] - get<1>(Data[id])) / get<2>(Data[id]), 2);
    test << get<0>(Data[id]) << " " << Predictions[id][0] << " " << get<1>(Data[id]) << " " << truth[id] << " " << get<2>(Data[id]) << std::ndl;
  }

  chi2 /= n;
  std::cout << "Final chi2 = " << chi2 << std::endl;
  std::cout << "Number of parameters = "<< np << std::endl;
  std::cout << "\n";


  delete nn;
  return 0;
}