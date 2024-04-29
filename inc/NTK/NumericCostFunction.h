//
// Author: Rabah Abdul Khalek - rabah.khalek@gmail.com
//

#pragma once

// Ceres solver
#include <ceres/ceres.h>

// NNAD
#include "NNAD/FeedForwardNN.h"

// Typedef for the data
typedef std::vector<double> dvec;

typedef std::tuple<dvec, dvec, dvec> Datapoint;

typedef std::vector<Datapoint> vecdata;

struct NumericCostFunction
{
  ~NumericCostFunction();

  NumericCostFunction(int const &,
                      vecdata const &,
                      std::vector<int> const &,
                      int const &);

  bool operator()(double const *const *, double *) const;

  double GetResidual(const int &) const;
  std::vector<double> GetResiduals() const;

  nnad::FeedForwardNN<double> *_nn;
  int _Np;
  vecdata _Data;
  std::vector<int> _NNarchitecture;
  int _Seed;
  int _ndata;
  int _OutSize;
};
