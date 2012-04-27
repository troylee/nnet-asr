#ifndef _CUSPARSE_LINEARITY_H_
#define _CUSPARSE_LINEARITY_H_


#include "cuComponent.h"
#include "cumatrix.h"


#include "Matrix.h"
#include "Vector.h"


namespace TNet {

  class CuSparseLinearity : public CuUpdatableComponent
  {
    public:

      CuSparseLinearity(size_t nInputs, size_t nOutputs, CuComponent *pPred); 
      ~CuSparseLinearity();  
      
      ComponentType GetType() const;
      const char* GetName() const;

      void PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y);
      void BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y);

      void Update();
      void UpdateMask();

      void ReadFromStream(std::istream& rIn);
      void WriteToStream(std::ostream& rOut);

      void L1(BaseFloat l1) {
        mL1Const = l1;
      }

    protected:
      CuMatrix<BaseFloat> mLinearity;  ///< Matrix with neuron weights
      CuVector<BaseFloat> mBias;       ///< Vector with biases
      CuMatrix<BaseFloat> mSparsityMask; ///< Mask which selects active weights

      CuMatrix<BaseFloat> mLinearityCorrection; ///< Matrix for linearity updates
      CuVector<BaseFloat> mBiasCorrection;      ///< Vector for bias updates

      CuMatrix<BaseFloat> mLinearityCorrectionAccu; ///< Accumulator for linearity updates

      BaseFloat mL1Const; ///< L1 regularization constant

      size_t mNFrames; ///< Number of accumulated frames 
      BaseFloat mSparsifyWeightThreshold; ///< Cutoff
      BaseFloat mUnsparsifyAccu; ///< Threshold to unsparsify the Cutoff

      
  };




  ////////////////////////////////////////////////////////////////////////////
  // INLINE FUNCTIONS 
  // CuSparseLinearity::
  inline 
  CuSparseLinearity::
  CuSparseLinearity(size_t nInputs, size_t nOutputs, CuComponent *pPred)
    : CuUpdatableComponent(nInputs, nOutputs, pPred), 
      mLinearity(nInputs,nOutputs), mBias(nOutputs), mSparsityMask(nInputs,nOutputs),
      mLinearityCorrection(nInputs,nOutputs), mBiasCorrection(nOutputs),
      mLinearityCorrectionAccu(nInputs,nOutputs),
      mNFrames(0), mSparsifyWeightThreshold(1.0e-3),
      mUnsparsifyAccu(1e20f)
  { 
    mLinearityCorrection.SetConst(0.0f);
    mBiasCorrection.SetConst(0.0f);
    mLinearityCorrectionAccu.SetConst(0.0f);
  }


  inline
  CuSparseLinearity::
  ~CuSparseLinearity()
  { }

  inline CuComponent::ComponentType
  CuSparseLinearity::
  GetType() const
  {
    return CuComponent::SPARSE_LINEARITY;
  }

  inline const char*
  CuSparseLinearity::
  GetName() const
  {
    return "<sparselinearity>";
  }



} //namespace



#endif
