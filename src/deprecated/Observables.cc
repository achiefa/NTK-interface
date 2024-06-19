#include <NTK/Observables.h>
#include <NNAD/FeedForwardNN.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace NTK
{
  typedef Eigen::Tensor<double, 1, Eigen::ColMajor> t1cm;
  typedef Eigen::Tensor<double, 2, Eigen::ColMajor> t2cm;

  // Implemented observables
  ObservableObjectFactory::RegisterObject("O1", O1::Create);
  ObservableObjectFactory::RegisterObject("Object_2", dO1::Create);
  
  
  /**
   * @note Highly coupled with Eigen tensor?
   */
  class O1 : public IObservableObject<t1cm>
  {
    public:
      O1 (nnad::FeedForwardNN<double>* NN) : _nn(NN) 
      {
        // Initialise the tensor
        _observable = TensorType(NN->GetArchitecture().back());
        _observable.setZero();
      }

      static IObservableObject* Create(nnad::FeedForwardNN<double>* NN) { return new O1(NN); }
    
    private:
      bool _evaluate(std::vector<double> input) {
        _observable = Eigen::TensorMap<TensorType> (input.data(), _observable.dimension(1)); // This part is repeated
        return true;
      }
      nnad::FeedForwardNN<double> *_nn;
  };


  class dO1 : public IObservableObject
  {
    public:
      dO1 (nnad::FeedForwardNN<double>* NN) : _nn(NN) 
      {
        // Initialise the tensor
        size_t nout = NN->GetArchitecture().back();
        size_t np = NN->GetParameterNumber();
        _observable = TensorType(np, nout);
        _observable.setZero();
      }
      static IObservableObject* Create(nnad::FeedForwardNN<double>* NN) { return new dO1(NN); }
    
    private:
      bool _evaluate(std::vector<double> input) {
        size_t nout = _observable.dimension(1);
        size_t np = _observable.dimension(2);
        // .data() is needed because returns a direct pointer to the memory array used internally by the vector
        Eigen::TensorMap<t2cm> temp (_nn->Derive(input).data(), nout, np + 1); // Col-Major

        // Get rid of the first column (the outputs) and stores only first derivatives
        Eigen::array<Eigen::Index, 2> offsets = {0, 1};
        Eigen::array<Eigen::Index, 2> extents = {nout, np};
        _observable = temp.slice(offsets, extents);
        return true;
      }
      nnad::FeedForwardNN<double> *_nn;
  };
}