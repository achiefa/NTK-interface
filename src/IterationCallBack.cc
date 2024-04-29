#include "NTK/AnalyticCostFunction.h"
#include "NTK/IterationCallBack.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <yaml-cpp/yaml.h>

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

    struct stat buffer;

    if (stat(OutputFolder.c_str(), &buffer) != 0)
        system(("mkdir " + OutputFolder).c_str());

    // Create log folder
    std::string LogFolder = OutputFolder + "/log/";
    if (stat(LogFolder.c_str(), &buffer) != 0)
        system(("mkdir " + OutputFolder + "/log/").c_str());
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

    nnad::FeedForwardNN<double> *NN = _chi2t->GetNN();

    // Store value as row major order
    int ndata = _chi2t->GetData().size();
    int Size = 4;
    int StepSize = int(ndata / 4);
    nnad::Matrix NTK {Size, Size, std::vector<double> (Size * Size, 0.)};
    for (int a = 0; a < Size; a++){
      for (int b = 0; b < Size; b++){
          std::vector<double> input_a = std::get<0>(_chi2t->GetData()[(a) * StepSize]);
          std::vector<double> input_b = std::get<0>(_chi2t->GetData()[(b) * StepSize]);
          NTK.SetElement(a,b, NN->NTK(input_a, input_b).GetElement(0,0));
      }
    }

    //NTK.Display();

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
    emitter << YAML::Key << "NTK" << YAML::Value << YAML::Flow << NTK.GetVector();
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