#ifndef _CU_RBM_H_
#define _CU_RBM_H_


#include "cuComponent.h"
#include "cumatrix.h"


#include "Matrix.h"
#include "Vector.h"


namespace TNet {

  class CuRbmBase : public CuUpdatableComponent
  {
   public:
    typedef enum {
      BERNOULLI,
      GAUSSIAN
    } RbmUnitType;
   
    CuRbmBase(size_t nInputs, size_t nOutputs, CuComponent *pPred) :
      CuUpdatableComponent(nInputs, nOutputs, pPred)
    { }
   
    
    virtual void Propagate(
      const CuMatrix<BaseFloat>& visProbs, 
      CuMatrix<BaseFloat>& hidProbs
    ) = 0;
    virtual void Reconstruct(
      const CuMatrix<BaseFloat>& hidState, 
      CuMatrix<BaseFloat>& visProbs
    ) = 0;
    virtual void RbmUpdate(
      const CuMatrix<BaseFloat>& pos_vis, 
      const CuMatrix<BaseFloat>& pos_hid, 
      const CuMatrix<BaseFloat>& neg_vis, 
      const CuMatrix<BaseFloat>& neg_hid
    ) = 0;

    virtual RbmUnitType VisType() = 0;
    virtual RbmUnitType HidType() = 0;
  };


  class CuRbm : public CuRbmBase
  {
    public:

      CuRbm(size_t nInputs, size_t nOutputs, CuComponent *pPred); 
      ~CuRbm();  
      
      ComponentType GetType() const;
      const char* GetName() const;

      //CuUpdatableComponent API
      void PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y);
      void BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y);

      void Update();

      //RBM training API
      void Propagate(const CuMatrix<BaseFloat>& visProbs, CuMatrix<BaseFloat>& hidProbs);
      void Reconstruct(const CuMatrix<BaseFloat>& hidState, CuMatrix<BaseFloat>& visProbs);
      void RbmUpdate(const CuMatrix<BaseFloat>& pos_vis, const CuMatrix<BaseFloat>& pos_hid, const CuMatrix<BaseFloat>& neg_vis, const CuMatrix<BaseFloat>& neg_hid);

      RbmUnitType VisType()
      { return mVisType; }

      RbmUnitType HidType()
      { return mHidType; }

      //static void BinarizeProbs(const CuMatrix<BaseFloat>& probs, CuMatrix<BaseFloat>& states);

      //I/O
      void ReadFromStream(std::istream& rIn);
      void WriteToStream(std::ostream& rOut);

    protected:
      CuMatrix<BaseFloat> mVisHid;  ///< Matrix with neuron weights
      CuVector<BaseFloat> mVisBias;       ///< Vector with biases
      CuVector<BaseFloat> mHidBias;       ///< Vector with biases

      CuMatrix<BaseFloat> mVisHidCorrection; ///< Matrix for linearity updates
      CuVector<BaseFloat> mVisBiasCorrection;      ///< Vector for bias updates
      CuVector<BaseFloat> mHidBiasCorrection;      ///< Vector for bias updates

      CuMatrix<BaseFloat> mBackpropErrBuf;

      RbmUnitType mVisType;
      RbmUnitType mHidType;

  };




  ////////////////////////////////////////////////////////////////////////////
  // INLINE FUNCTIONS 
  // CuRbm::
  inline 
  CuRbm::
  CuRbm(size_t nInputs, size_t nOutputs, CuComponent *pPred)
    : CuRbmBase(nInputs, nOutputs, pPred), 
      mVisHid(nInputs,nOutputs), 
      mVisBias(nInputs), mHidBias(nOutputs),
      mVisHidCorrection(nInputs,nOutputs), 
      mVisBiasCorrection(nInputs), mHidBiasCorrection(nOutputs),
      mBackpropErrBuf(),
      mVisType(BERNOULLI),
      mHidType(BERNOULLI)
  { 
    mVisHidCorrection.SetConst(0.0);
    mVisBiasCorrection.SetConst(0.0);
    mHidBiasCorrection.SetConst(0.0);
  }


  inline
  CuRbm::
  ~CuRbm()
  { }

  inline CuComponent::ComponentType
  CuRbm::
  GetType() const
  {
    return CuComponent::RBM;
  }

  inline const char*
  CuRbm::
  GetName() const
  {
    return "<rbm>";
  }



} //namespace



#endif
