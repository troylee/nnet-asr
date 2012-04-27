#ifndef _BIASED_LINEARITY_H_
#define _BIASED_LINEARITY_H_


#include "Component.h"

#include "Matrix.h"
#include "Vector.h"


namespace TNet {

class BiasedLinearity : public UpdatableComponent
{
 public:

  BiasedLinearity(size_t nInputs, size_t nOutputs, Component *pPred);
  ~BiasedLinearity() { } 
  
  ComponentType GetType() const
  { return BIASED_LINEARITY; }

  const char* GetName() const
  { return "<BiasedLinearity>"; }

  Component* Clone() const;

  void PropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y);
  void BackpropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y);

  void ReadFromStream(std::istream& rIn);
  void WriteToStream(std::ostream& rOut);

  /// calculate gradient
  void Gradient();
  /// accumulate gradient from other components
  void AccuGradient(const UpdatableComponent& src, int thr, int thrN);  
  /// update weights, reset the accumulator
  void Update(int thr, int thrN);

 protected:
  Matrix<BaseFloat> mLinearity;  ///< Matrix with neuron weights
  Vector<BaseFloat> mBias;       ///< Vector with biases

  const Matrix<BaseFloat>* mpLinearity;
  const Vector<BaseFloat>* mpBias;

  Matrix<BaseFloat> mLinearityCorrection; ///< Matrix for linearity updates
  Vector<BaseFloat> mBiasCorrection;      ///< Vector for bias updates

  Matrix<double> mLinearityCorrectionAccu; ///< Matrix for summing linearity updates
  Vector<double> mBiasCorrectionAccu;      ///< Vector for summing bias updates

};




////////////////////////////////////////////////////////////////////////////
// INLINE FUNCTIONS 
// BiasedLinearity::
inline 
BiasedLinearity::
BiasedLinearity(size_t nInputs, size_t nOutputs, Component *pPred)
  : UpdatableComponent(nInputs, nOutputs, pPred), 
    mLinearity(), mBias(), //cloned instaces don't need this
    mpLinearity(&mLinearity), mpBias(&mBias), 
    mLinearityCorrection(nInputs,nOutputs), mBiasCorrection(nOutputs),
    mLinearityCorrectionAccu(), mBiasCorrectionAccu() //cloned instances don't need this
{ }

inline
Component* 
BiasedLinearity::
Clone() const
{
  BiasedLinearity* ptr = new BiasedLinearity(GetNInputs(), GetNOutputs(), NULL);
  ptr->mpLinearity = mpLinearity; //copy pointer from currently active weights
  ptr->mpBias = mpBias;           //...

  ptr->mLearningRate = mLearningRate;
  
  return ptr;
}



} //namespace



#endif
