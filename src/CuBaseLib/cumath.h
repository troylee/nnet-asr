#ifndef _CUMATH_H_
#define _CUMATH_H_

#include "cumatrix.h"

#include "Timer.h"
#include "cudevice.h"

namespace TNet {
  
  
  /**
   * Group of Math operations for the NN training
   */
  template<typename _ElemT>
  class CuMath 
  {
   public:

    /// Y = Sigmoid(X)
    static void Sigmoid(CuMatrix<_ElemT>& Y, const CuMatrix<_ElemT>& X)
    { Error("__func__ Not implemented"); }

    /// Eout = E(1-E) * Y
    static void DiffSigmoid(CuMatrix<_ElemT>& Eout, const CuMatrix<_ElemT>& Ein, const CuMatrix<_ElemT>& Y)
    { Error("__func__ Not implemented"); }

    /// Y = Softmax(X)
    static void Softmax(CuMatrix<_ElemT>& Y, const CuMatrix<_ElemT>& X)
    { Error("__func__ Not implemented"); }

    /// for DCT in FeaCat
    static void BlockLinearity(CuMatrix<_ElemT>& Y, const CuMatrix<_ElemT>& X, const CuMatrix<_ElemT>& block_transf)
    { Error("__func__ Not implemented"); }

    static void Expand(CuMatrix<_ElemT>& Y, const CuMatrix<_ElemT>& X, const CuVector<int>& frameOffsets)
    { Error("__func__ Not implemented"); }

    /// ie. switch cols according to copyFrom
    static void Rearrange(CuMatrix<_ElemT>& Y, const CuMatrix<_ElemT>& X, const CuVector<int>& copyFrom)
    { Error("__func__ Not implemented"); }

    /// ie. switch rows according to copyFrom   
    static void Randomize(CuMatrix<_ElemT>& Y, const CuMatrix<_ElemT>& X, const CuVector<int>& copyFrom)
    { Error("__func__ Not implemented"); }

    /// check match in the classification for Xentropy
    static void CheckClass(const CuMatrix<_ElemT>& out, const CuMatrix<_ElemT> &des, CuVector<int>& match)
    { Error("__func__ Not implemented"); }
    
    /// gemm with offset for CuSharedLinearity
    static void OffsetGemm(char transA, char transB, _ElemT alpha, const CuMatrix<_ElemT>& A, const CuMatrix<_ElemT>& B, _ElemT beta, CuMatrix<_ElemT>& C, int offA, int offB, int offC)
    { Error("__func__ Not implemented"); }

    /// gemv with offset for CuRecurrent
    static void OffsetGemv(char trans, _ElemT alpha, const CuMatrix<_ElemT>& A, const _ElemT* x, size_t dimX, _ElemT beta, _ElemT* y, size_t dimY, size_t offsetY)
    { Error("__func__ Not implemented"); }

    /// ger for weight updates in CuRecurrent
    static void BlasGer(_ElemT alpha, const _ElemT* x, size_t dimX, const _ElemT* y, size_t dimY, CuMatrix<_ElemT>& A)
    { Error("__func__ Not implemented"); }

    /// concatenate one vector several times for CuSharedLinearity
    static void VecExpand(const CuVector<_ElemT>&in, CuVector<_ElemT>&out)
    { Error("__func__ Not implemented"); }

    /// sum the vector as if it was matrix data for CuSharedLinearity
    static void VecAddColSum(_ElemT alpha, const CuVector<_ElemT>&in, _ElemT beta, CuVector<_ElemT>&out)
    { Error("__func__ Not implemented"); }

  }; //class CuMath::


  //////////////////////////////////////////////////////////////////////////////
  //// CuMath<> Template specializations (float)
  ////
  template<>
  void CuMath<float>::Sigmoid(CuMatrix<float>& Y, const CuMatrix<float>& X);

  template<>
  void CuMath<float>::DiffSigmoid(CuMatrix<float>& Eout, const CuMatrix<float>& Ein, const CuMatrix<float>& Y);
    
  template<>
  void CuMath<float>::Softmax(CuMatrix<float>& Y, const CuMatrix<float>& X);

  template<>
  void CuMath<float>::BlockLinearity(CuMatrix<float>& Y, const CuMatrix<float>& X, const CuMatrix<float>& block_transf);

  template<>
  void CuMath<float>::Expand(CuMatrix<float>& Y, const CuMatrix<float>& X, const CuVector<int>& frameOffsets);

  template<>
  void CuMath<float>::Rearrange(CuMatrix<float>& Y, const CuMatrix<float>& X, const CuVector<int>& copyFrom);

  template<>
  void CuMath<float>::Randomize(CuMatrix<float>& Y, const CuMatrix<float>& X, const CuVector<int>& copyFrom);

  template<>
  void CuMath<float>::CheckClass(const CuMatrix<float>& out, const CuMatrix<float> &des, CuVector<int>& match);

  template<>
  void CuMath<float>::OffsetGemm(char transA, char transB, float alpha, const CuMatrix<float>& A, const CuMatrix<float>& B, float beta, CuMatrix<float>& C, int offA, int offB, int offC);

  template<>
  void CuMath<float>::OffsetGemv(char trans, float alpha, const CuMatrix<float>& A, const float* x, size_t dimX, float beta, float* y, size_t dimY, size_t offsetY);

  template<>
  void CuMath<float>::BlasGer(float alpha, const float* x, size_t dimX, const float* y, size_t dimY, CuMatrix<float>& A);

  template<>
  void CuMath<float>::VecExpand(const CuVector<float>&in, CuVector<float>&out);

  template<>
  void CuMath<float>::VecAddColSum(float alpha, const CuVector<float>&in, float beta, CuVector<float>&out);


  //////////////////////////////////////////////////////////////////////////////
  //// CuMath<> Template specializations (double)
  ////
  template<>
  void CuMath<double>::Sigmoid(CuMatrix<double>& Y, const CuMatrix<double>& X);

  template<>
  void CuMath<double>::DiffSigmoid(CuMatrix<double>& Eout, const CuMatrix<double>& Ein, const CuMatrix<double>& Y);
    
  template<>
  void CuMath<double>::Softmax(CuMatrix<double>& Y, const CuMatrix<double>& X);

  template<>
  void CuMath<double>::BlockLinearity(CuMatrix<double>& Y, const CuMatrix<double>& X, const CuMatrix<double>& block_transf);

  template<>
  void CuMath<double>::Expand(CuMatrix<double>& Y, const CuMatrix<double>& X, const CuVector<int>& frameOffsets);

  template<>
  void CuMath<double>::Rearrange(CuMatrix<double>& Y, const CuMatrix<double>& X, const CuVector<int>& copyFrom);

  template<>
  void CuMath<double>::Randomize(CuMatrix<double>& Y, const CuMatrix<double>& X, const CuVector<int>& copyFrom);

  template<>
  void CuMath<double>::CheckClass(const CuMatrix<double>& out, const CuMatrix<double> &des, CuVector<int>& match);

}

#endif
