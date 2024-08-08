#include <vector>
#include <string>

#include <NNAD/FeedForwardNN.h>

namespace NTK
{
  /**
   * @brief List of implemented observables
   * 
   * @todo Maybe this will be moved into a separate
   * part or changed...
   */
  enum ObservableNames: int {O2, O3};
  typedef std::string obs;

  // This class is meant to be constructed only one, in train.cc before training starts.
  // This is why there is an UpdateNetwork method.
  class ObservableHandler
  {
    public:
      // The first constructor does not need to provide a list of observable,
      // whereas the second one needs a list of string names
      ObservableHandler(nnad::FeedForwardNN<double> *NN);
      ObservableHandler(nnad::FeedForwardNN<double> *NN, obs ObsName) : ObservableHandler(NN)
      {
        _CheckAndAdd(ObsName);
      }

      ObservableHandler(nnad::FeedForwardNN<double> *NN, std::vector<obs> ObsNames) : ObservableHandler(NN)
      {
        for names in ObsName
          _CheckAndAdd(ObsName);
      }
      
      void UpdateNetwork(nnad::FeedForwardNN<double> *NN) {_nn = NN;}


      void AddObservable(obs ObsName)
      {
        _CheckAndAdd(ObsName);
      }

      /**
       * @brief Compute the actual observables
       * @todo Maybe this is where I should introduce lazy evaluation, namely
       * actually evaluate the elements when they are called
       * 
       */
      void ComputeObservables(data, stepsize)

    
      auto GetObservable(ObsName /*????*/)
      {
        _ComputeElements(); // Compute only if not already done.
        return Observable->Evaluate(Elelements)
      }

    private:
    void _CheckObservableIsImplemented(obs ObsName);
    void _CheckAndAdd(obs ObsName); // Functional programming in full glory
    void _AddIfNotPresent(Element);

    // I think this is the tricky part
    void _ComputeElements(vector of elements) // This function will first check whether elements have already been computed and stored in some private attribute
    {

    }
    nnad::FeedForwardNN<double> *_nn;
    std::vector<obs> _Observables;
    std::map<std::string, Element> _Elements;
  };






  // Does not depend on the NN, but only on the elements (right?)
  template <type ObsType>
  class Observable
  {
    public:
      Observable(/*???*/)
      {
        // Here we can initialise the Eigen constructions
      };

      ObsType Evaluate(); // This will depend on the actual implementation.
    
    private:
      boh _MapAPIinEigenOperations();
      std::vector<ElementsObjects> Ingredients_;
      type_boh Combination_;


  };







  // For instance d_NN or dd_NN
  // Dependencies
  // - NN
  // 
  class IElementObject
  {
    // Ensures derived class call
    // the correct destructor (i.e., top of the chain)
    virtual ~IElementObject() {}

    // Implement here the lazy evaluation. Only compute the element when actually called
    IElementObject (nnad::FeedForwardNN<double> NN)
    {
      // Initialise tensor that will contain the Element, e.g.
      // ```
      //  Eigen::Tensor<double, 3> d_NN (Size, Nout, Np);
      //  d_NN.setZero();
      // ```
    }

    // Actually, this can also be the operator () instead, or both
    virtual void Evaluate (input_a, index a)
    {
      // For instance
      // ```
      //  Eigen::TensorMap< Eigen::Tensor<double, 2, Eigen::ColMajor> > temp (NN->Derive(input_a).data(), Nout, Np + 1); // Col-Major
      //  Eigen::array<Eigen::Index, 2> offsets = {0, 1};
      //  Eigen::array<Eigen::Index, 2> extents = {Nout, Np};
      //  d_NN.chip(a,0) = temp.slice(offsets, extents); ------> not sure with this, since to fill d_NN we will need the dependence on an external action (in this case may be _ComputeElements) and on an external index, that is `a`. Is it possible to decouple
      // ```
      // The problem is that this guy is strongly cuopled 
    }

    private:
      //

  };

  class IElementObject
  {
    // Ensures derived class call
    // the correct destructor (i.e., top of the chain)
    virtual ~IElementObject() {}

    // Actually, this can also be the operator () instead, or both
    virtual void Evaluate (input_a, index a) = 0;
  
  };

}