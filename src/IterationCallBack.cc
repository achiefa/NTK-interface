#include "NTK/AnalyticCostFunction.h"
#include "NTK/IterationCallBack.h"
#include "NTK/NumericalDerivative.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <yaml-cpp/yaml.h>

// Eigen tensors
#include <unsupported/Eigen/CXX11/Tensor>

namespace NTK
{
  volatile sig_atomic_t stop;

//_________________________________________________________________________
void inthand(int signum)
{
  stop = 1;
}

//_________________________________________________________________________
IterationCallBack::IterationCallBack(bool VALIDATION,
                                    std::string OutputFolder,
                                    int replica,
                                    std::vector<double*> const& Parameters,
                                    AnalyticCostFunction *chi2t,
                                    AnalyticCostFunction *chi2v):
    _VALIDATION(VALIDATION),
    _OutputFolder(OutputFolder),
    _replica(replica),
    _BestIteration(0),
    _Bestchi2v(1e10),
    _Parameters(Parameters),
    _chi2t(chi2t),
    _chi2v(chi2v)
{
    signal(SIGINT, inthand);
    const int Np = _Parameters.size();
    _BestParameters.resize(Np);
    for (int ip = 0; ip < Np; ip++)
        _BestParameters[ip] = _Parameters[ip][0];

    // Create output folder but throw an exception if it does not
    // exist.
    }

    //_________________________________________________________________________
  ceres::CallbackReturnType IterationCallBack::operator()(const ceres::IterationSummary &summary)
  {
    // Return if the iteration is not succesful
    if (!summary.step_is_successful)
      return ceres::SOLVER_CONTINUE;

    // Get NN parameters
    int Np = _Parameters.size();
    std::vector<double> vpar(Np);

    for (int ip = 0; ip < Np; ip++)
      vpar[ip] = _Parameters[ip][0];

    double chi2v = 0;
    if (_VALIDATION)
      {
        _chi2v->SetParameters(vpar);
        chi2v = _chi2v->Evaluate();
        if (chi2v <= _Bestchi2v)
          {
            _Bestchi2v = chi2v;
            _BestIteration = summary.iteration;
            _BestParameters = vpar;
          }
      }

    // Training chi2's
    _chi2t->SetParameters(vpar);
    const double chi2t_tot = _chi2t->Evaluate();

    //________________________________________________________________________________
    // |====================================|
    // |             Obserables             |
    // |====================================|
    // Dependencies for independet environment
    // nnad::FeedForwardNN<double> *NN
    // Nout
    nnad::FeedForwardNN<double> *NN = _chi2t->GetNN();
    int Nout = NN->GetArchitecture().back();
    std::vector<double> parameters = NN->GetParameters();

    // Store value as row major order
    // TO-DO
    // I would rather prefer to read the step size from the config
    // parser, or conceive something more general and not hard-coded.
    // Also the calculation of `StepSize` is not free from errors and
    // undesired outcomes.
    int ndata = _chi2t->GetData().size();
    int Size = 4;
    int StepSize = int(ndata / 4);
    const double eps = 1.e-5;

    Eigen::Tensor<double, 3> d_NN (Size, Nout, Np);
    Eigen::Tensor<double, 4> dd_NN (Size, Nout, Np, Np);
    d_NN.setZero();
    dd_NN.setZero();

    for (int a = 0; a < Size; a++){
      std::vector<double> input_a = std::get<0>(_chi2t->GetData()[(a) * StepSize]);

      // -------------------------- First derivative -------------------------
      // .data() is needed because returns a direct pointer to the memory array used internally by the vector
      std::vector<double> DD = NN->Derive(input_a);
      Eigen::TensorMap< Eigen::Tensor<double, 2, Eigen::ColMajor> > temp (DD.data(), Nout, Np + 1); // Col-Major

      // Get rid of the first column (the outputs) and stores only first derivatives
      Eigen::array<Eigen::Index, 2> offsets = {0, 1};
      Eigen::array<Eigen::Index, 2> extents = {Nout, Np};
      d_NN.chip(a,0) = temp.slice(offsets, extents);


      // -------------------------- Second derivative -------------------------
      std::vector<double> results_vec = NTK::helper::HelperSecondFiniteDer(NN, input_a, eps); // Compute second derivatives

      // Store into ColMajor tensor
      // The order of the dimensions has been tested in "SecondDerivative", and worked out by hand.
      Eigen::TensorMap< Eigen::Tensor<double, 3, Eigen::ColMajor> > ddNN (results_vec.data(), Nout, Np + 1, Np);

      // Swap to ColMajor for compatibility and reshape
      Eigen::array<int, 3> new_shape{{0, 2, 1}};
      Eigen::Tensor<double, 3> ddNN_reshape = ddNN.shuffle(new_shape);

      // Get rid of the first column (the firs derivatives) and stores only second derivatives
      Eigen::array<Eigen::Index, 3> offsets_3 = {0, 0, 1};
      Eigen::array<Eigen::Index, 3> extents_3 = {Nout, Np, Np};
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
    //_____________________________________________

    // Output parameters into yaml file
    YAML::Emitter emitter;
    emitter << YAML::BeginSeq;
    emitter << YAML::Flow << YAML::BeginMap;
    emitter << YAML::Key << "iteration" << YAML::Value << summary.iteration;
    //emitter << YAML::Key << "training chi2" << YAML::Value << 2 * summary.cost / _chi2t->GetDataPointNumber();
    emitter << YAML::Key << "training chi2" << YAML::Value << chi2t_tot;
    //emitter << YAML::Key << "training partial chi2s" << YAML::Value << YAML::Flow << chi2t_par;
    if (_VALIDATION)
      emitter << YAML::Key << "validation chi2" << YAML::Value << chi2v;
    //emitter << YAML::Key << "parameters" << YAML::Value << YAML::Flow << vpar;
    emitter << YAML::Key << "NTK_eigen" << YAML::Value << YAML::Flow << NTK_Eigen_vec;
    emitter << YAML::Key << "O_3" << YAML::Value << YAML::Flow << O3_vec;
    emitter << YAML::EndMap;
    emitter << YAML::EndSeq;
    emitter << YAML::Newline;
    std::ofstream fout(_OutputFolder + "/log/replica_" + std::to_string(_replica) + ".yaml", std::ios::out | std::ios::app);
    fout << emitter.c_str();
    fout.close();

    // Manual stop by the user
    if (stop)
      return ceres::SOLVER_TERMINATE_SUCCESSFULLY;

    return ceres::SOLVER_CONTINUE;
  }
}