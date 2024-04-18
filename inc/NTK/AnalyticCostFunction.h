//
//
//

#pragma once

// Ceres solver
#include <ceres/ceres.h>

// NNAD
#include "NNAD/FeedForwardNN.h"
#include <yaml-cpp/yaml.h>

#include <fstream>

namespace NTK
{
  // Typedef for data
  // The first element is the input
  // The second element is the output + noise
  // The third element is the absolute value of the noise
  // Note that each of them can be an array.
  typedef std::vector<double> dvec;

  typedef std::tuple<dvec, dvec, dvec> Datapoint;

  typedef std::vector<Datapoint> vecdata;

  class AnalyticCostFunction : public ceres::CostFunction
  {
  public:
    AnalyticCostFunction(nnad::FeedForwardNN<double> *NN, vecdata const &Data);
    //virtual ~AnalyticCostFunction();

    double GetResidual(const int &) const;
    std::vector<double> GetResiduals() const;

    // Analytic chi2 in ceres
    virtual bool Evaluate(double const *const *, double *, double **) const;
    double Evaluate() const;

    void SetParameters(std::vector<double> const &pars) { _nn->SetParameters(pars); }

    // Numeric chi2 in ceres
    bool operator()(double const *const *, double *) const;

    vecdata GetData() const { return _Data; };

    nnad::FeedForwardNN<double>* GetNN() const { return _nn; };
  
  private:
     nnad::FeedForwardNN<double> *_nn;
     int _Np;
     vecdata _Data;
     int _ndata;
     int _OutSize;
     //std::vector<int> _NNarchitecture;
     //int _Seed;
  };
}