#ifndef _NETWORK_H_
#define _NETWORK_H_

#include "Component.h"
#include "BiasedLinearity.h"
#include "SharedLinearity.h"
#include "Activation.h"

#include "Vector.h"

#include <vector>


namespace TNet {

class Network
{
//////////////////////////////////////
// Typedefs
typedef std::vector<Component*> LayeredType;
  
  //////////////////////////////////////
  // Disable copy construction and assignment
 private:
  Network(Network&); 
  Network& operator=(Network&);
   
 public:
  // allow incomplete network creation
  Network() 
  { }

  ~Network();

  int Layers() const
  { return mNnet.size(); }

  Component& Layer(int i)
  { return *mNnet[i]; }
 
  const Component& Layer(int i) const
  { return *mNnet[i]; }

  /// Feedforward the data per blocks, this needs less memory, 
  /// and allows to process very long files.
  /// It does not trim the *_frm_ext, but uses it 
  /// for concatenation of segments
  void Feedforward(const Matrix<BaseFloat>& in, Matrix<BaseFloat>& out, 
                   size_t start_frm_ext, size_t end_frm_ext);
  /// forward the data to the output
  void Propagate(const Matrix<BaseFloat>& in, Matrix<BaseFloat>& out);
  /// backpropagate the error while calculating the gradient
  void Backpropagate(const Matrix<BaseFloat>& globerr); 

  /// accumulate the gradient from other networks
  void AccuGradient(const Network& src, int thr, int thrN);
  /// update weights, reset the accumulator
  void Update(int thr, int thrN);
  
  Network* Clone(); ///< Clones the network

  void ReadNetwork(const char* pSrc);     ///< read the network from file
  void ReadNetwork(std::istream& rIn);    ///< read the network from stream
  void WriteNetwork(const char* pDst);    ///< write network to file
  void WriteNetwork(std::ostream& rOut);  ///< write network to stream

  size_t GetNInputs() const; ///< Dimensionality of the input features
  size_t GetNOutputs() const; ///< Dimensionality of the desired vectors

  void SetLearnRate(BaseFloat learnRate); ///< set the learning rate value
  BaseFloat GetLearnRate();  ///< get the learning rate value

  void SetWeightcost(BaseFloat l2); ///< set the L2 regularization const

  void ResetBunchsize(); ///< reset the frame counter (needed for L2 regularization
  void AccuBunchsize(const Network& src); ///< accumulate frame counts in bunch (needed in L2 regularization

 private:
  /// Creates a component by reading from stream
  Component* ComponentFactory(std::istream& In);
  /// Dumps component into a stream
  void ComponentDumper(std::ostream& rOut, Component& rComp);

 private:
  LayeredType mNnet; ///< container with the network layers

};
  

//////////////////////////////////////////////////////////////////////////
// INLINE FUNCTIONS 
// Network::
inline Network::~Network() {
  //delete all the components
  LayeredType::iterator it;
  for(it=mNnet.begin(); it!=mNnet.end(); ++it) {
    delete *it;
  }
}

    
inline size_t Network::GetNInputs() const {
  assert(mNnet.size() > 0);
  return mNnet.front()->GetNInputs();
}


inline size_t
Network::
GetNOutputs() const
{
  assert(mNnet.size() > 0);
  return mNnet.back()->GetNOutputs();
}



inline void
Network::
SetLearnRate(BaseFloat learnRate)
{
  LayeredType::iterator it;
  for(it=mNnet.begin(); it!=mNnet.end(); ++it) {
    if((*it)->IsUpdatable()) {
      dynamic_cast<UpdatableComponent*>(*it)->LearnRate(learnRate);
    }
  }
}


inline BaseFloat
Network::
GetLearnRate()
{
  //TODO - learn rates may differ layer to layer
  assert(mNnet.size() > 0);
  for(size_t i=0; i<mNnet.size(); i++) {
    if(mNnet[i]->IsUpdatable()) {
      return dynamic_cast<UpdatableComponent*>(mNnet[i])->LearnRate();
    }
  }
  Error("No updatable NetComponents");
  return -1;
}


inline void
Network::
SetWeightcost(BaseFloat l2)
{
  LayeredType::iterator it;
  for(it=mNnet.begin(); it!=mNnet.end(); ++it) {
    if((*it)->IsUpdatable()) {
      dynamic_cast<UpdatableComponent*>(*it)->Weightcost(l2);
    }
  }
}


inline void 
Network::
ResetBunchsize()
{
  LayeredType::iterator it;
  for(it=mNnet.begin(); it!=mNnet.end(); ++it) {
    if((*it)->IsUpdatable()) {
      dynamic_cast<UpdatableComponent*>(*it)->Bunchsize(0);
    }
  }
}

inline void
Network::
AccuBunchsize(const Network& src)
{
  assert(Layers() == src.Layers());
  assert(Layers() > 0);

  for(int i=0; i<Layers(); i++) {
    if(Layer(i).IsUpdatable()) {
      UpdatableComponent& tgt_comp = dynamic_cast<UpdatableComponent&>(Layer(i));
      const UpdatableComponent& src_comp = dynamic_cast<const UpdatableComponent&>(src.Layer(i));
      tgt_comp.Bunchsize(tgt_comp.Bunchsize()+src_comp.GetOutput().Rows());
    }
  }
}

  

} //namespace

#endif


