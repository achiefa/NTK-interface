// NTK
#include "NTK/derive.h"

// NNAD
#include "NNAD/FeedForwardNN.h"

#include <cstdio>
#include <filesystem>


int main()
{
    // Define architecture
  const std::vector<int> arch{1, 2, 2};

  // Initialise NN
  const nnad::FeedForwardNN<double> nn{arch, 0, nnad::OutputFunction::QUADRATIC, true};

  // Input vector
  std::vector<double> x{0.1};
  std::cout << "Input: x = { ";
  for (auto const& e : x)
    std::cout << e << " ";
  std::cout << "}\n" << std::endl;

  // Get NN at x
  std::vector<double> adNN = nn.Derive(x);

  //Create std::function for derivatives
  std::function<std::vector<double>(std::vector<double> const&, std::vector<double>)> dNN
  {
    [&] (std::vector<double> const& x, std::vector<double> parameters) -> std::vector<double>
    {
        nnad::FeedForwardNN<double> aux_nn{nn};
        aux_nn.SetParameters(parameters);

        return aux_nn.Derive(x);
    }
  };

  const double eps = 1e-5;
  std::vector<double> parameters = nn.GetParameters();
  const int np = nn.GetParameterNumber();
  std::vector<std::vector<double>> results = NTK::FiniteDifference(dNN, parameters, x, eps);
 
  std::printf("Output :\n");
  for (auto &out : adNN)
    std::printf("%.5f \n", out);

  for (int mu = 0; mu < np; mu++) {
    for (int nu = 0; nu < np + 1; nu++) {
        std::printf("_________________\n");
        if (nu == 0){
            std::cout << "######################" << std::endl;
            std::cout << "Analytic derivatives : " << adNN[0 + (mu + 1) * arch.back()] << std::endl;
            std::cout << "Analytic derivatives : " << adNN[1 + (mu + 1) * arch.back()] << std::endl;
        }
        for(int i = 0; i < arch.back(); i++) {
            if (nu == 0)
              std::printf("d_%.1d phi_%.1d : %.5f\n", mu + 1, i, results[mu][i + (nu) * arch.back()]);
            else
              // This is the function to be used if one wants to neglect the first derivatives
              std::printf("d_%.1d d_%.1d phi_%.1d : %.5f\n", mu + 1, nu, i, results[mu][i + (nu) * arch.back()]);
        }
        if (nu == 0)
            std::cout << "######################" << std::endl;
    }
  }
  return 0;
}