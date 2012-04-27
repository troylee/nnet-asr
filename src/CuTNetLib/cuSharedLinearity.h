#ifndef _CUSHARED_LINEARITY_H_
#define _CUSHARED_LINEARITY_H_


#include "cuComponent.h"
#include "cumatrix.h"


#include "Matrix.h"
#include "Vector.h"


namespace TNet {

  class CuSharedLinearity : public CuUpdatableComponent
  {
    public:

      CuSharedLinearity(size_t nInputs, size_t nOutputs, CuComponent *pPred); 
      ~CuSharedLinearity();  
      
      ComponentType GetType() const;
      const char* GetName() const;

      void PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y);
      void BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y);

      void Update();

      void ReadFromStream(std::istream& rIn);
      void WriteToStream(std::ostream& rOut);

    protected:
      CuMatrix<BaseFloat> mLinearity;  ///< Matrix with neuron weights
      CuVector<BaseFloat> mBias;       ///< Vector with biases

      CuMatrix<BaseFloat> mLinearityCorrection; ///< Matrix for linearity updates
      CuVector<BaseFloat> mBiasCorrection;      ///< Vector for bias updates

      int mNInstances;
      CuVector<BaseFloat> mBiasExpand;
      CuVector<BaseFloat> mBiasCorrectionExpand;

  };




  ////////////////////////////////////////////////////////////////////////////
  // INLINE FUNCTIONS 
  // CuSharedLinearity::
  inline 
  CuSharedLinearity::
  CuSharedLinearity(size_t nInputs, size_t nOutputs, CuComponent *pPred)
    : CuUpdatableComponent(nInputs, nOutputs, pPred), 
      mNInstances(0)
  { }


  inline
  CuSharedLinearity::
  ~CuSharedLinearity()
  { }

  inline CuComponent::ComponentType
  CuSharedLinearity::
  GetType() const
  {
    return CuComponent::SHARED_LINEARITY;
  }

  inline const char*
  CuSharedLinearity::
  GetName() const
  {
    return "<sharedlinearity>";
  }



} //namespace



#endif
