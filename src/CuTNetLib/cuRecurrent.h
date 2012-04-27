#ifndef _CU_RECURRENT_H_
#define _CU_RECURRENT_H_


#include "cuComponent.h"
#include "cumatrix.h"


#include "Matrix.h"
#include "Vector.h"


namespace TNet {

  class CuRecurrent : public CuUpdatableComponent
  {
    public:

      CuRecurrent(size_t nInputs, size_t nOutputs, CuComponent *pPred); 
      ~CuRecurrent();  
      
      ComponentType GetType() const;
      const char* GetName() const;

      //CuUpdatableComponent API
      void PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y);
      void BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y);

      void Update();

      //Recurrent training API
      void BpttOrder(int ord) {
        mBpttOrder = ord;
        mInputHistory.Init(ord+1,GetNInputs()+GetNOutputs());
      }
      void ClearHistory() {
        mInputHistory.SetConst(0.0);
        if(mOutput.MSize() > 0) {
          mOutput.SetConst(0.0);
        }
      }

      //I/O
      void ReadFromStream(std::istream& rIn);
      void WriteToStream(std::ostream& rOut);

    protected:
      CuMatrix<BaseFloat> mLinearity;
      CuVector<BaseFloat> mBias;

      CuMatrix<BaseFloat> mLinearityCorrection;
      CuVector<BaseFloat> mBiasCorrection;

      CuMatrix<BaseFloat> mInputHistory;

      int mBpttOrder;
  };




  ////////////////////////////////////////////////////////////////////////////
  // INLINE FUNCTIONS 
  // CuRecurrent::
  inline 
  CuRecurrent::
  CuRecurrent(size_t nInputs, size_t nOutputs, CuComponent *pPred)
    : CuUpdatableComponent(nInputs, nOutputs, pPred), 
      mLinearity(nInputs+nOutputs,nOutputs),
      mBias(nOutputs),
      mLinearityCorrection(nInputs+nOutputs,nOutputs), 
      mBiasCorrection(nOutputs)
  { }


  inline
  CuRecurrent::
  ~CuRecurrent()
  { }

  inline CuComponent::ComponentType
  CuRecurrent::
  GetType() const
  {
    return CuComponent::RECURRENT;
  }

  inline const char*
  CuRecurrent::
  GetName() const
  {
    return "<recurrent>";
  }



} //namespace



#endif
