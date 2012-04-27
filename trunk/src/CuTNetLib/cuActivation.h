
#ifndef _CUACT_FUN_I_
#define _CUACT_FUN_I_


#include "cuComponent.h"
#include "cumatrix.h"


namespace TNet
{

  /**
   * Common interface for activation functions
   */
  class CuActivation : public CuComponent 
  {
    public:
      CuActivation(size_t nInputs, size_t nOutputs, CuComponent *pPred);

    protected:
  };


  /**
   * Sigmoid activation function
   */
  class CuSigmoid : public CuActivation
  {
    public:
      CuSigmoid(size_t nInputs, size_t nOutputs, CuComponent *pPred);

      ComponentType GetType() const;
      const char* GetName() const;

    protected:
      void PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y);
      void BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y);
  };
    

  /**
   * Softmax activation function
   */
  class CuSoftmax : public CuActivation
  {
    public:
      CuSoftmax(size_t nInputs, size_t nOutputs, CuComponent *pPred);

      ComponentType GetType() const;
      const char* GetName() const;

    protected:
      void PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y);
      void BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y);
  };


  //////////////////////////////////////////////////////////////////////////
  // Inline functions
  // Activation::
  inline
  CuActivation::
  CuActivation(size_t nInputs, size_t nOutputs, CuComponent *pPred)
    : CuComponent(nInputs,nOutputs, pPred)
  { 
    assert(nInputs == nOutputs);
  } 


  //////////////////////////////////////////////////////////////////////////
  // Inline functions
  // Sigmoid::
  inline
  CuSigmoid::
  CuSigmoid(size_t nInputs, size_t nOutputs, CuComponent *pPred)
    : CuActivation(nInputs,nOutputs, pPred) 
  { } 

  inline CuComponent::ComponentType
  CuSigmoid::
  GetType() const
  {
    return CuComponent::SIGMOID;
  }

  inline const char*
  CuSigmoid::
  GetName() const
  {
    return "<sigmoid>";
  }



  //////////////////////////////////////////////////////////////////////////
  // Inline functions
  // Softmax::
  inline
  CuSoftmax::
  CuSoftmax(size_t nInputs, size_t nOutputs, CuComponent *pPred)
    : CuActivation(nInputs,nOutputs, pPred) 
  { } 

  inline CuComponent::ComponentType
  CuSoftmax::
  GetType() const
  {
    return CuComponent::SOFTMAX;
  }

  inline const char*
  CuSoftmax::
  GetName() const
  {
    return "<softmax>";
  }


} //namespace


#endif
