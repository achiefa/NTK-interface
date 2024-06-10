#include <cxxabi.h>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <NNAD/FeedForwardNN.h>
#pragma once

namespace NTK {
typedef std::vector<double> data;
typedef nnad::FeedForwardNN<double> NNAD;
template <int _RANK>
using Tensor = Eigen::Tensor<double, _RANK>;

/**
 * @brief Type demangler
 *
 * The type name returned by the `std::typeinfo::name()` call is the mangled
 * type, which is the internal name that the compiler uses to identify types.
 * The demangler convert it into a human-readable name (demangler). Note that
 * the C string returned by `abi::__cxa_demangle` must be deallocated by
 * the caller using `free`, as mentioned in the abi documentation:
 * https://gcc.gnu.org/onlinedocs/libstdc++/libstdc++-html-USERS-4.3/a01696.html.
 *
 * here `typeid` is an operator and not function. It is directly implemented
 * in the compiler, and for that reason it can accept a typename as an argument.
 *
 * @tparam D type
 * @return std::string
 */
template <typename D>
std::string type_demangler() {
  int r;
  std::string name;
  char* mangled_name = abi::__cxa_demangle(typeid(D).name(), 0, 0, &r);
  name += mangled_name;
  std::free(mangled_name);
  return name;
}

/**
 * @brief IObservable
 *
 * This is the common interface for all types of observables. It serves
 * as base class template meant for CRTP, and the first template argument
 * represents the derived class. This base class can also be interpreted as
 * a wrapper to the Eigen::Tensor class. The latter is a class template, and
 * in particular it requires the specification of the tensor rank at compile
 * time. Hence, `IObservable` inherits this dependency as well, and introduces
 * another template argument which specifies the rank of the tensor representing
 * the observable.
 *
 * Eigen::Tensor does not require compile-time information to set the size of
 * each dimension of the tensor once the rank is known. For this reason, the
 * constructor is a variadic template that allows multiple entries defining the
 * size of each dimension. The variadic private method `_initialise_tensor`
 * initialises the entries of the tensor to zero. The tensor is then stored
 * into a private attribute that can be accessed through the getter `GetTensor`
 *
 * Despite being defined static in the base class, the actual definition depends
 * on the template parameters. This, the derived classes don't need to define
 * their own ids since the base class takes care of that. Moreover, being
 * `id_base3` a static variable, one can access the type id of a class without
 * actually instantiating an object.
 *
 * @tparam D : The derived class used in the CRTP
 * @tparam _RANK : The rank of the tensor that stores the observable
 * @todo The variadic constructor should only accept integers variables
 * but no check is applied.
 */
template <typename D, int _RANK>
class IObservable {
 public:
  template <typename... Args>
  IObservable(Args&&... args) {
    _initialise_tensor(args...);
    _d = _tensor.dimensions();
  }

  Tensor<_RANK> GetTensor() { return _tensor; }
  std::string GetID() const { return id; }
  static const std::string id;

 protected:
  typename Tensor<_RANK>::Dimensions  _d;
  Tensor<_RANK>                       _tensor;

 private:
  template <typename... Args>
  void _initialise_tensor(Args&&... args) {
    Tensor<_RANK> temp(std::forward<Args>(args)...);
    temp.setZero();
    _tensor = temp;
    return;
  }
  D* derived() { return static_cast<D*>(this); }
  template <typename>
  static std::string static_id() {
    return type_demangler<D>();
  }
};
template <typename D, int _RANK>
const std::string IObservable<D, _RANK>::id = IObservable::static_id<D>();

/**
 * @brief Basic decorator
 *
 * Decorate the IObservable class. It keeps the same signature for
 * the template arguments so that it can be used as a base class
 * in the CRTP.
 *
 * @tparam D: The derived class used in the CRTP
 * @tparam _RANK: The rank of the tensor that stores the observable
 */
template <typename D, int _RANK>
class BASIC : public IObservable<D, _RANK> {
 public:
  void Evaluate(const data& X, int a, NNAD* nn) {
    if (a > this->_d[0] - 1)
      throw std::invalid_argument("The size you provided is greater than the actual size of the tensor.");
    this->_tensor.chip(a, 0) = static_cast<D*>(this)->algorithm_impl(X, a, nn);
    data_map[a] = X;
  }


  /**
   * @brief Evaluate specifying data and nn
   * 
   * @param Xv 
   * @param nn 
   */
  void Evaluate(const std::vector<data>& Xv, NNAD* nn) {
    this->_tensor.setZero();
    for (size_t a = 0; a < Xv.size(); a++) {
      Evaluate(Xv[a], a, nn);
    }
  }

  /**
   * @brief Evaluate using stored data and nn
   */
  void Evaluate() {
    if (_nn == nullptr || !(_data_loaded))
      throw std::invalid_argument("Either the network or the data have not been set.");
    this->_tensor.setZero();
    for (size_t a = 0; a < data_map.size(); a++) {
      Evaluate(data_map[a], a, _nn);
    }
  }

  void Evaluate(NNAD* nn) {
    if (!(_data_loaded))
      throw std::invalid_argument("Data has not been set.");
    this->_tensor.setZero();
    for (size_t a = 0; a < data_map.size(); a++) {
      Evaluate(data_map[a], a, nn);
    }
  }

  void Evaluate(std::vector<data>& Xv) {
    if (_nn == nullptr)
      throw std::invalid_argument("Network has not been set.");
    this->_tensor.setZero();
    for (size_t a = 0; a < Xv.size(); a++) {
      Evaluate(Xv[a], a, _nn);
    }
  }

  bool is_computed() { return data_map.size() == this->_d[0]; }

  std::map<int, data> GetDataMap() { return data_map; }

 protected:
  template <typename... Args> BASIC(Args&&... args) : IObservable<D, _RANK>(args...), _nn() {}
  template <typename... Args> BASIC(NNAD* nn, Args&&... args) : IObservable<D, _RANK>(args...),  _nn(nn) {}
  template <typename... Args> BASIC(std::vector<data> data_batch, NNAD* nn, Args&&... args) : IObservable<D, _RANK>(args...), _nn(nn) {
    // Fill data map
    for(size_t i=0; i < data_batch.size(); i++)
      data_map[i] = data_batch[i];
    _data_loaded = true;
  }

  ~BASIC() {}
  std::map<int, data> data_map;
  NNAD*               _nn;
  bool                _data_loaded = false;
};

/**
 * @brief Combined decorator
 * 
 * @tparam D 
 * @tparam _RANK 
 * @todo The `check_observables` function only checks that the single basic
 * observables have been computed for the entire data batch. However, this
 * does not check that the data batch is the same, and hence must be
 * implemented (and tested as well).
 * @todo I should implement a check in the in template arguments for the
 * `Evaluate` function.
 */
template <typename D, int _RANK>
class COMBINED : public IObservable<D, _RANK> {
 public:
  template <typename... Obs>
  void Evaluate(Obs*... obs) {
    if (!check_observables(obs...))
      throw std::logic_error("Something");
    this->_tensor = static_cast<D*>(this)->contract_impl(obs...);
  }

  template <typename... Obs>
  Tensor<_RANK> contract_impl(Obs*... obs) {
    Tensor<_RANK> t;
    return t;
  }

 protected:
  template <typename... Args>
  COMBINED(Args&&... args) : IObservable<D, _RANK>(args...) {}
  ~COMBINED() {}

 private:
  template <typename... Obs>
  Tensor<_RANK> contract(Obs*... obs) {
    return static_cast<D*>(this)->contract_impl(obs ...);
  };

  template <typename... Obs>
  bool check_observables(Obs*... obs) {
    // Using a fold expression
    bool check = true;
    ([&] {
      if (!obs->is_computed())
        check = false;
    } () , ...);
    return check;
  }
};


class dNN : public BASIC<dNN, 3> {
  public:
  /**
   * @brief Construct a new d N N object
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