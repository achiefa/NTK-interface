#include <vector>
#include <string>

#include <NNAD/FeedForwardNN.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace NTK
{
  /**
   * @brief Observable factory
   * 
   * Extensible factory implementation.
   * From Mike Shah's talk at CppCon 2021.
   * I want to be able to create different type at runtime. For
   * instance, an user (i.e, the client) may be interested in
   * defining her observable. For that reason I'm not using
   * 'enum class'.
   * 
   * The idea is that the user can use this function using the
   * following API
   * ```
   *  ObservableObjectFactory::RegisterObject("Object_1", Object_1::Create);
   *  ObservableObjectFactory::RegisterObject("Object_2", Object_2::Create);
   *  
   *  std::vector<IObservableObject*>  ObservableCollection // This may become a class
   * 
   *  // Loop over a list of object to be created
   *  for (auto& type : tpyes) {
   *    ObservableCollection.push_back(ObservableObjectFactory::CreateSignleObject(type))
   *  }
   * 
   * To make everything work, the objects must have the following class structure
   * class Object_1 : public IObservableObject{
   *    public:
   *      Object_1( ... ) {...}
   *      .
   *      .  Some methods
   *      .
   *      static IObservableObject* create() {
   *        return new Object_1( ... );
   *        }
   * }
   * 
   * @todo I think all static member functions could be moved into class objects...
   * 
   * @note From https://www.learncpp.com/cpp-tutorial/static-member-functions/?utm_content=cmp-true
   * `member functions defined inside the class definition are implicitly inline. 
   * Member functions defined outside the class definition are not implicitly inline,
   * but can be made inline by using the inline keyword. Therefore a static member
   * function that is defined in a header file should be made inline so as not to
   * violate the One Definition Rule (ODR) if that header is then included into 
   * multiple translation units.`
   */
  //template<typename otype>
  class ObservableObjectFactory{
    public:
      typedef nnad::FeedForwardNN<double> *NN;
      
      // Here we are first deferencing `CreateObjectCallBack`
      // which is a pointer, through the deference syntax
      //  *pointer_to_object.
      // Then I evaluate the function pointed to by `CreateObjectCallBack`
      //  (*pointer_to_object)().
      // Finally, the return of this function is again a pointer, and we
      // want to derefenrece it again
      //  *(*pointer_to_object)().
      // NOTE: This typedef means that anytime I call
      // *(*CreateObjectCallBack)(), it returns an IObservableObject.
      // CreateObjectCallBack is just a function that returns a pointer
      // to whatever the object is.
      // See https://stackoverflow.com/questions/4295432/typedef-function-pointer
      // This typedef is equivalent to
      // CreateObjectCallBack cb;    --> compile equally as IObservableObject* (*cb)(NN)
      // which is a typedef for a function pointer
      typedef IObservableObject* (*CreateObjectCallBack)(NN);

      // Register a new user created object type. The key component
      // si the ability to `register` and `unregister` object types.
      // Types will be stored in a std::map.
      // The `CreateObjectCallBack` is a function that tells us
      // how we crate whatever the type is. This means that anytime we
      // want to register an object, we need to provide the name of the object
      // and how to create it.
      inline static void RegisterObject(const std::string& type, CreateObjectCallBack cb) {
        s_Objects[type] = cb;
      }

      // Unregister a user created object type
      // Remove from the map
      inline static void UnregisterObject(const std::string type) {
        s_Objects.erase(type);
      }

      // Factory method
      // This factory method loops over the map of
      // registered objects
      inline static IObservableObject* CreateObject(const std::string& type, NN nn) {
        CallBackMap::iterator it = s_Objects.find(type); // Random access data structure could be adopted to make this part faster 
        if (it != s_Objects.end()) {
          // Call the CallBack function
          // Returns what CreateObjectCallBack returns, that is the pointer to the Observable;
          return (it->second)(nn); 
        }
        return nullptr;
      }

      // destructor something along the line
      /* ~factory()
      {
        auto it = _function_map.begin();
        for(it ; it != _function_map.end() ; ++it)
        {
          delete (*it).second;
        } */

    private:
      // Convenience typedef
      typedef std::map<std::string, CreateObjectCallBack> CallBackMap;

      // Map of all different objects that we can create
      inline static CallBackMap s_Objects;
  };


  class ObservableCollection{
    public:
    typedef IObservableObject* (*ObjectEvaluation)(std::vector<double>);
    typedef std::vector<IObservableObject*> ObservableCollection;

      ObservableCollection(std::vector<double> data_batch){  }
      ObservableCollection(std::vector<double> data_batch){ _SetBatch(data_batch); }

      inline static void AddObservable(const std::string &obs){
        // Check if type already included
        if (std::find(_ObservablesList.begin(), _ObservablesList.end(), obs) != _ObservablesList.end()){
          _ObservablesList.push_back(obs);
        }
        // Here would be coll to add
        // else{
        //    log.debug("try to add an observable previously listed.")
        //  }
      }
      inline static void AddObservable(std::vector<std::string> &obs){
        for (auto &ob : obs)
          AddObservable(ob);
      };

      void ComputeObservables()
      {
        if (_check_empty_batch())
          std::cerr << "No data batch provided" << std::endl;

        // Check common basic operators
        _check_common_basic_observables();
        _initialise_tensors();

        // Compute common observables
        for (auto &input : _data_batch){
          
        }
      }

    private:
      bool _check_empty_batch() {return _data_batch.empty();}
      // Set data_batch
      inline void _SetBatch(std::vector<double> data_batch) { 
        _data_batch = data_batch;
        _data_batch_size = data_batch.size();
        }
      inline static void _CreateObjects(ObservableObjectFactory::NN nn){
        for (auto &obs : _ObservablesList){
          _IObjectCollection[obs] = ObservableObjectFactory::CreateObject(obs, nn);
        }
      }
      void _check_common_operators();
      inline static std::vector<double> _data_batch;
      inline static int _data_batch_size;
      inline static std::map<std::string, IObservableObject*> _IObjectCollection;
      inline static std::map<std::string, std>
      inline static std::vector<std::string> _ObservablesList;
      inline static std::vector<std::string> _CommonElementList;
  };


  template <typename rank, typename Derived>
  class IObservableObject
  {
    public:
      IObservableObject() {_initialise_tensor();} 
      virtual ~IObservableObject() {} // Ensures derived class call the correct destructor (i.e., top of the chain)
      static inline void EvaluateOnBatch() {}


    private:
      EigenTensor _observable;
      nnad::FeedForwardNN<double>* _nn;
      Derived& _derived;

      virtual void _evaluate();
      void _initialise_tensor() { _observable = Eigen }
      void _deduce_derived() { T }
  };


  


}
