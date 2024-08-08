#include "NTK/Observable.h"
#include "NNAD/FeedForwardNN.h"
#include "NTK/utility_test.hpp"

#include "benchmark/benchmark.h"

#include <thread>


void BlockSecondDerivative(std::function<std::vector<double>(std::vector<double> const&, std::vector<double>)> f,
                          std::vector<double> parameters, double eps, NTK::data x,  
                          std::size_t N, std::size_t start_row, std::size_t end_row) {
  std::vector<double> results (N * N, 0.);               
  for (int ip = start_row; ip < end_row; ip++) {
    auto start = std::chrono::high_resolution_clock::now();
      for (int jp = ip; jp < N; jp++) {
        std::vector<double> ParFF = parameters;
        std::vector<double> ParFB = parameters;
        std::vector<double> ParBF = parameters;
        std::vector<double> ParBB = parameters;
        ParFF[ip] +=  parameters[ip] * eps;
        ParFF[jp] +=  parameters[jp] * eps;
        ParFB[ip] +=  parameters[ip] * eps;
        ParFB[jp] -=  parameters[jp] * eps;
        ParBF[ip] -=  parameters[ip] * eps;
        ParBF[jp] +=  parameters[jp] * eps;
        ParBB[ip] -=  parameters[ip] * eps;
        ParBB[jp] -=  parameters[jp] * eps;
        std::vector<double> f_FF = f(x, ParFF);
        std::vector<double> f_FB = f(x, ParFB);
        std::vector<double> f_BF = f(x, ParBF);
        std::vector<double> f_BB = f(x, ParBB);
        std::vector<double> f_FF_BB (f_FF.size(), 0.);
        std::vector<double> f_FB_BF (f_FF.size(), 0.);
        std::vector<double> res (f_FF.size(), 0.);
        std::transform(f_FF.begin(), f_FF.end(), f_BB.begin(), f_FF_BB.begin(), [&] (double f1, double f2) {return f1 + f2;});
        std::transform(f_FB.begin(), f_FB.end(), f_BF.begin(), f_FB_BF.begin(), [&] (double f1, double f2) {return f1 + f2;});
        std::transform(f_FF_BB.begin(), f_FF_BB.end(), f_FB_BF.begin(), results.begin(), [&] (double f1, double f2)
        {
          return (f1 - f2) / 4. / eps / eps / parameters[ip] / parameters[jp];
        });
      }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Time elapsed is " << elapsed_seconds.count() << std::endl;
  } 
}

//____________________________________________________________________
void third_derivative_bench(benchmark::State &s) {

  // Size of the network
  int N = s.range(0);
  const std::vector<int> arch{1, N, N, 1};

  // Initialise network and wrapper
  std::unique_ptr<NTK::NNAD> nn = std::make_unique<NTK::NNAD>(arch, 0, nnad::OutputFunction::QUADRATIC, false);
  int nout = arch.back();
  int np = nn->GetParameterNumber();

  std::function<std::vector<double>(std::vector<double> const&, std::vector<double>)> NNAD_wrapper
  {
    [&] (std::vector<double> const& x, std::vector<double> parameters) -> std::vector<double>
    {
      nnad::FeedForwardNN<double> aux_nn{*(nn.get())}; // Requires pointer dereference
      aux_nn.SetParameters(parameters);
      return aux_nn.Evaluate(x);
    }
  };
  double eps = 1.e-5;

  // Initialise data batch
  int batch_size = 4;
  std::vector<NTK::data> data_batch (batch_size);
  data_batch[0] = {0.0};
  data_batch[1] = {0.25};
  data_batch[2] = {0.50};
  data_batch[3] = {0.75};

  // Setting up for threads
  std::size_t num_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  size_t n_rows = np / num_threads;

  for(auto _ : s) {
    size_t end_row = 0;
    for (size_t i = 0; i < num_threads -1; i++) {
      auto start_row = i * n_rows;
      end_row = start_row + n_rows;
      threads.emplace_back(
        [&] { 
          BlockSecondDerivative(NNAD_wrapper, nn->GetParameters(), eps, data_batch[3], np, start_row, end_row);
        });
    }

    // Now we need to wait for all threads to complete
    for (auto &t : threads) t.join();
    // Clear the threads each iteration of the benchmark
    threads.clear();
  }
}

BENCHMARK(third_derivative_bench)
  //->Arg(2)
  //->Arg(5)
  ->Arg(20)
  //->Arg(20)
  ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();