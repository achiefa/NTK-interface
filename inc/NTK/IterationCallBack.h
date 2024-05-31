//
//
//
//

#pragma once

// Cerse Sovler
#include <ceres/ceres.h>

#include <yaml-cpp/yaml.h>

// Standard libs
#include <iostream>
#include <math.h>
#include <string>

namespace NTK
{
    class IterationCallBack : public ceres::IterationCallback
    {
    public:
        IterationCallBack(bool VALIDATION,
                          std::string OutputFolder,
                          int replica,
                          std::vector<double*> const& Parameters,
                          AnalyticCostFunction *chi2t,
                          AnalyticCostFunction *chi2v = nullptr);

        virtual ceres::CallbackReturnType operator()(const ceres::IterationSummary &);

        int GetBestIteratoin() { return _BestIteration; };
        double GetBestValidationChi2() { return _Bestchi2v; };
        std::vector<double> GetBestParameters() { return _BestParameters; };

    private:
        const bool _VALIDATION;
        const std::string _OutputFolder;
        const int _replica;
        int _BestIteration;
        double _Bestchi2v;
        std::vector<double> _BestParameters;
        std::vector<double*> _Parameters;
        AnalyticCostFunction *_chi2t;
        AnalyticCostFunction *_chi2v;                  
    };
}