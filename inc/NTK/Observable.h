#include <cxxabi.h>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <NNAD/FeedForwardNN.h>

#include "NTK/Observable_int.h"
#pragma once

namespace NTK {

  class dNN : public BASIC<dNN, 3> {
    public:
    /**
     * @brief Construct a new d NN object
     * 
     * @param batch_size 
     * @param nout 
     * @param np 
     */
    dNN(int batch_size, int nout, int np);

    /**
     * @brief Construct a new d NN object
     * 
     * Requires network and batch_size. The Observable is automatically
     * coupled to the network is is initialised with. The other dimensions
     * (such as the number of output nodes and parameters) are deduced from
     * the network.
     * 
     * @param nn 
     * @param batch_size
     */
    dNN(NNAD* nn, int batch_size);

    /**
     * @brief Construct a new dNN object
     * 
     * Requires the network and the data_batch. All the other parameters are
     * deduced.
     * @param nn 
     * @param data_batch 
     */
    dNN(NNAD* nn, std::vector<data> data_batch);

    private:
    Tensor<2> algorithm_impl(const data&, int , NNAD*);
    friend BASIC<dNN,3>;
  };


  class ddNN : public BASIC<ddNN, 4> {
  public:
    /**
     * @brief Construct a new ddNN object
     * 
     * @param batch_size 
     * @param nout 
     * @param np 
     */
    ddNN(int batch_size, int nout, int np);

    /**
     * @brief Construct a new ddNN object
     * 
     * Requires network and batch_size. The Observable is automatically
     * coupled to the network is is initialised with. The other dimensions
     * (such as the number of output nodes and parameters) are deduced from
     * the network.
     * 
     * @param nn 
     * @param batch_size
     */
    ddNN(NNAD* nn, int batch_size);

    /**
     * @brief Construct a new ddNN object
     * 
     * Requires the network and the data_batch. All the other parameters are
     * deduced.
     * @param nn 
     * @param data_batch 
     */
    ddNN(NNAD* nn, std::vector<data> data_batch);

  private:
    Tensor<3> algorithm_impl(const data&, int, NNAD*);
    friend BASIC<ddNN,4>;
  };


  class d3NN : public BASIC<d3NN, 5> {
  public:
    /**
     * @brief Construct a new ddNN object
     *
     * @param batch_size
     * @param nout
     * @param np
     */
    d3NN(int batch_size, int nout, int np);

    /**
     * @brief Construct a new ddNN object
     *
     * Requires network and batch_size. The Observable is automatically
     * coupled to the network is is initialised with. The other dimensions
     * (such as the number of output nodes and parameters) are deduced from
     * the network.
     *
     * @param nn
     * @param batch_size
     */
    d3NN(NNAD* nn, int batch_size);

    /**
     * @brief Construct a new ddNN object
     *
     * Requires the network and the data_batch. All the other parameters are
     * deduced.
     * @param nn
     * @param data_batch
     */
    d3NN(NNAD* nn, std::vector<data> data_batch);

  private:
    Tensor<4> algorithm_impl(const data&, int, NNAD*);
    friend BASIC<d3NN,5>;
  };


  class O2 : public COMBINED<O2, 4> {
  public:
    O2(int, int);

  private:
    Tensor<4> contract_impl(dNN*);
    friend COMBINED<O2, 4>;
  };


  class O3 : public COMBINED<O3, 6> {
  public:
    O3(int, int);

  private:
    Tensor<6> contract_impl(dNN*, ddNN*);
    friend COMBINED<O3, 6>;
  };


  class O4 : public COMBINED<O4, 8> {
  public:
    O4(int, int);

  private:
    Tensor<8> contract_impl(dNN*, ddNN*, d3NN*);
    friend COMBINED<O4, 8>;
  };

}  // namespace NTK