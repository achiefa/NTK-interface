#include "NTK/AnalyticCostFunction.h"
#include "NTK/IterationCallBack.h"
#include "NTK/Timer.h"
#include "NTK/direxists.h"

#include <getopt.h>
#include <sys/stat.h>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include "NTK/NumericalDerivative.h"


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

  //________________________________________________________________________________
    // |====================================|
    // |             Obserables             |
    // |====================================|
    // Dependencies for independet environment
    // nnad::FeedForwardNN<double> *NN
    // Nout
    int Nout = nn->GetArchitecture().back();
    std::vector<double> parameters = nn->GetParameters();

    // Store value as row major order
    // TO-DO
    // I would rather prefer to read the step size from the config
    // parser, or conceive something more general and not hard-coded.
    // Also the calculation of `StepSize` is not free from errors and
    // undesired outcomes.
    int Size = 4;
    int StepSize = int(ndata / 4);
    const double eps = 1.e-5;

    Eigen::Tensor<double, 3> d_NN (Size, Nout, np);
    Eigen::Tensor<double, 4> dd_NN (Size, Nout, np, np);
    d_NN.setZero();
    dd_NN.setZero();
    std::cout << np << std::endl;
    std::cout << Nout << std::endl;

    for (int a = 0; a < Size; a++){
      std::vector<double> input_a = std::get<0>(Data[(a) * StepSize]);

      // -------------------------- First derivative -------------------------
      // .data() is needed because returns a direct pointer to the memory array used internally by the vector
      std::vector<double> DD = nn->Derive(input_a);
      Eigen::TensorMap< Eigen::Tensor<double, 2, Eigen::ColMajor> > temp (DD.data(), Nout, np + 1); // Col-Major

      // Get rid of the first column (the outputs) and stores only first derivatives
      Eigen::array<Eigen::Index, 2> offsets = {0, 1};
      Eigen::array<Eigen::Index, 2> extents = {Nout, np};
      d_NN.chip(a,0) = temp.slice(offsets, extents);


      // -------------------------- Second derivative -------------------------
      std::vector<double> results_vec = NTK::helper::HelperSecondFiniteDer(nn, parameters, input_a, np, Nout, eps); // Compute second derivatives

      // Store into ColMajor tensor
      // The order of the dimensions has been tested in "SecondDerivative", and worked out by hand.
      Eigen::TensorMap< Eigen::Tensor<double, 3, Eigen::ColMajor> > ddNN (results_vec.data(), Nout, np + 1, np);

      // Swap to ColMajor for compatibility and reshape
      Eigen::array<int, 3> new_shape{{0, 2, 1}};
      Eigen::Tensor<double, 3> ddNN_reshape = ddNN.shuffle(new_shape);

      // Get rid of the first column (the firs derivatives) and stores only second derivatives
      Eigen::array<Eigen::Index, 3> offsets_3 = {0, 0, 1};
      Eigen::array<Eigen::Index, 3> extents_3 = {Nout, np, np};
      dd_NN.chip(a,0) = ddNN_reshape.slice(offsets_3, extents_3);
    }

    // Contract first derivatives to get the NTK
    Eigen::array<Eigen::IndexPair<int>, 1> double_contraction = { Eigen::IndexPair<int>(2,2) };
    Eigen::Tensor<double, 4> NTK_Eigen = d_NN.contract(d_NN, double_contraction);
    std::vector<double> NTK_Eigen_vec ( NTK_Eigen.data(),  NTK_Eigen.data() + NTK_Eigen.size() );

    // Contract first and second derivatives
    Eigen::array<Eigen::IndexPair<int>, 0> tensor_product = {  };
    Eigen::array<Eigen::IndexPair<int>, 2> first_contraction = { Eigen::IndexPair<int>(2,2), Eigen::IndexPair<int>(3,5) };
    Eigen::Tensor<double, 6> d_mu_d_nu_f_ia = d_NN.contract(d_NN, tensor_product);
    Eigen::Tensor<double, 6> O3 = dd_NN.contract(d_mu_d_nu_f_ia, first_contraction);
    std::vector<double> O3_vec ( O3.data(),  O3.data() + O3.size() );

    // Write to YAML
    // Output parameters into yaml file
    YAML::Emitter emitter;
    emitter << YAML::BeginSeq;
    emitter << YAML::Flow << YAML::BeginMap;
    emitter << YAML::Key << "iteration" << YAML::Value << int(0);
    emitter << YAML::Key << "NTK_eigen" << YAML::Value << YAML::Flow << NTK_Eigen_vec;
    emitter << YAML::Key << "O_3" << YAML::Value << YAML::Flow << O3_vec;
    emitter << YAML::EndMap;
    emitter << YAML::EndSeq;
    emitter << YAML::Newline;
    std::ofstream fout(FitFolder + "/log/replica_" + std::to_string(replica) + ".yaml", std::ios::out | std::ios::app);
    fout << emitter.c_str();
    fout.close();
    //_____________________________________________


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