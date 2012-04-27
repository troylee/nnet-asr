#ifndef _CUSHARED_LINEARITY_H_
#define _CUSHARED_LINEARITY_H_


#include "Component.h"

#include "Matrix.h"
#include "Vector.h"


namespace TNet {

class SharedLinearity : public UpdatableComponent
{
 public:
  SharedLinearity(size_t nInputs, size_t nOutputs, Component *pPred); 
  ~SharedLinearity();  
  
  ComponentType GetType() const 
  { return SHARED_LINEARITY; }

  const char* GetName() const
  { return "<SharedLinearity>"; }

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

  Matrix<BaseFloat>* mpLinearity;
  Vector<BaseFloat>* mpBias;

  Matrix<BaseFloat> mLinearityCorrection; ///< Matrix for linearity updates
  Vector<BaseFloat> mBiasCorrection;      ///< Vector for bias updates

  Matrix<double> mLinearityCorrectionAccu; ///< Accumulator for linearity updates
  Vector<double> mBiasCorrectionAccu;      ///< Accumulator for bias updates
  
  int mNInstances;
};




////////////////////////////////////////////////////////////////////////////
// INLINE FUNCTIONS 
// SharedLinearity::
inline 
SharedLinearity::
SharedLinearity(size_t nInputs, size_t nOutputs, Component *pPred)
  : UpdatableComponent(nInputs, nOutputs, pPred),
    mpLinearity(&mLinearity), mpBias(&mBias), 
    mNInstances(0)
{ }


inline
SharedLinearity::
~SharedLinearity()
{ }


inline
Component*
SharedLinearity::
Clone() const
{
  SharedLinearity* ptr = new SharedLinearity(GetNInputs(),GetNOutputs(),NULL);
  ptr->mpLinearity = mpLinearity;
  ptr->mpBias = mpBias;

  ptr->mLinearityCorrection.Init(mpLinearity->Rows(),mpLinearity->Cols());
  ptr->mBiasCorrection.Init(mpBias->Dim());

  ptr->mNInstances = mNInstances;

  ptr->mLearningRate = mLearningRate;


  return ptr;
}



} //namespace



#endif
