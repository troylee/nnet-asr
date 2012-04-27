#ifndef _CUBIASED_LINEARITY_H_
#define _CUBIASED_LINEARITY_H_


#include "cuComponent.h"
#include "cumatrix.h"


#include "Matrix.h"
#include "Vector.h"


namespace TNet {

  class CuBiasedLinearity : public CuUpdatableComponent
  {
    public:

      CuBiasedLinearity(size_t nInputs, size_t nOutputs, CuComponent *pPred); 
      ~CuBiasedLinearity();  
      
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

  };




  ////////////////////////////////////////////////////////////////////////////
  // INLINE FUNCTIONS 
  // CuBiasedLinearity::
  inline 
  CuBiasedLinearity::
  CuBiasedLinearity(size_t nInputs, size_t nOutputs, CuComponent *pPred)
    : CuUpdatableComponent(nInputs, nOutputs, pPred), 
      mLinearity(nInputs,nOutputs), mBias(nOutputs),
      mLinearityCorrection(nInputs,nOutputs), mBiasCorrection(nOutputs)
  { 
    mLinearityCorrection.SetConst(0.0);
    mBiasCorrection.SetConst(0.0);
  }


  inline
  CuBiasedLinearity::
  ~CuBiasedLinearity()
  { }

  inline CuComponent::ComponentType
  CuBiasedLinearity::
  GetType() const
  {
    return CuComponent::BIASED_LINEARITY;
  }

  inline const char*
  CuBiasedLinearity::
  GetName() const
  {
    return "<biasedlinearity>";
  }



} //namespace



#endif
