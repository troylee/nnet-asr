#ifndef _CU_RBM_SPARSE_H_
#define _CU_RBM_SPARSE_H_


#include "cuComponent.h"
#include "cumatrix.h"
#include "cuRbm.h"


#include "Matrix.h"
#include "Vector.h"


namespace TNet {

  class CuRbmSparse : public CuRbmBase
  {
    public:

      CuRbmSparse(size_t nInputs, size_t nOutputs, CuComponent *pPred); 
      ~CuRbmSparse();  
      
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

      ////// sparsity 
      BaseFloat mSparsityPrior; ///< sparsity target (unit activity prior)
      BaseFloat mLambda; ///< exponential decay factor for q (observed probability of unit to be active)
      BaseFloat mSparsityCost; ///< sparsity cost coef.

      CuVector<BaseFloat> mSparsityQ;
      CuVector<BaseFloat> mSparsityQCurrent;
      CuVector<BaseFloat> mVisMean; ///< buffer for mean visible

  };




  ////////////////////////////////////////////////////////////////////////////
  // INLINE FUNCTIONS 
  // CuRbmSparse::
  inline 
  CuRbmSparse::
  CuRbmSparse(size_t nInputs, size_t nOutputs, CuComponent *pPred)
    : CuRbmBase(nInputs, nOutputs, pPred), 
      mVisHid(nInputs,nOutputs), 
      mVisBias(nInputs), mHidBias(nOutputs),
      mVisHidCorrection(nInputs,nOutputs), 
      mVisBiasCorrection(nInputs), mHidBiasCorrection(nOutputs),
      mBackpropErrBuf(),
      mVisType(BERNOULLI),
      mHidType(BERNOULLI),

      mSparsityPrior(0.0001),
      mLambda(0.95),
      mSparsityCost(1e-7),
      mSparsityQ(nOutputs),
      mSparsityQCurrent(nOutputs),
      mVisMean(nInputs)
  { 
    mVisHidCorrection.SetConst(0.0);
    mVisBiasCorrection.SetConst(0.0);
    mHidBiasCorrection.SetConst(0.0);

    mSparsityQ.SetConst(mSparsityPrior);
    mSparsityQCurrent.SetConst(0.0);
    mVisMean.SetConst(0.0);
  }


  inline
  CuRbmSparse::
  ~CuRbmSparse()
  { }

  inline CuComponent::ComponentType
  CuRbmSparse::
  GetType() const
  {
    return CuComponent::RBM_SPARSE;
  }

  inline const char*
  CuRbmSparse::
  GetName() const
  {
    return "<rbmsparse>";
  }



} //namespace



#endif
