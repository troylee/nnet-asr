#ifndef TNet_Vector_cc
#define TNet_Vector_cc

#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include "Common.h"

#ifdef HAVE_ATLAS
extern "C"{
  #include <cblas.h>
}
#endif

#include "Common.h"
#include "Matrix.h"
#include "Vector.h"

namespace TNet
{

#ifdef HAVE_ATLAS
  template<>
     float
    BlasDot<>(const Vector<float>& rA, const Vector<float>& rB)
    {
      assert(rA.mDim == rB.mDim);
      return cblas_sdot(rA.mDim, rA.pData(), 1, rB.pData(), 1);
    }

  template<>
     double
    BlasDot<>(const Vector<double>& rA, const Vector<double>& rB)
    {
      assert(rA.mDim == rB.mDim);
      return cblas_ddot(rA.mDim, rA.pData(), 1, rB.pData(), 1);
    }

  template<>
     Vector<float>&
    Vector<float>::
    BlasAxpy(const float alpha, const Vector<float>& rV)
    {
      assert(mDim == rV.mDim);
      cblas_saxpy(mDim, alpha, rV.pData(), 1, mpData, 1);
      return *this;
    }

  template<>
     Vector<double>&
    Vector<double>::
    BlasAxpy(const double alpha, const Vector<double>& rV)
    {
      assert(mDim == rV.mDim);
      cblas_daxpy(mDim, alpha, rV.pData(), 1, mpData, 1);
      return *this;
    }

  template<>
     Vector<int>&
    Vector<int>::
    BlasAxpy(const int alpha, const Vector<int>& rV)
    {
      assert(mDim == rV.mDim);
      for(int i=0; i<Dim(); i++) {
        (*this)[i] += rV[i];
      }
      return *this;
    }


  template<>
     Vector<float>&
    Vector<float>::
    BlasGemv(const float alpha, const Matrix<float>& rM, MatrixTrasposeType trans, const Vector<float>& rV, const float beta)
    {
      assert((trans == NO_TRANS && rM.Cols() == rV.mDim && rM.Rows() == mDim)
          || (trans ==    TRANS && rM.Rows() == rV.mDim && rM.Cols() == mDim));

      cblas_sgemv(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(trans), rM.Rows(), rM.Cols(), alpha, rM.pData(), rM.Stride(),
                  rV.pData(), 1, beta, mpData, 1);
      return *this;
    }



  template<>
     Vector<double>&
    Vector<double>::
    BlasGemv(const double alpha, const Matrix<double>& rM, MatrixTrasposeType trans, const Vector<double>& rV, const double beta)
    {
      assert((trans == NO_TRANS && rM.Cols() == rV.mDim && rM.Rows() == mDim)
          || (trans ==    TRANS && rM.Rows() == rV.mDim && rM.Cols() == mDim));

      cblas_dgemv(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(trans), rM.Rows(), rM.Cols(), alpha, rM.pData(), rM.Stride(),
                  rV.pData(), 1, beta, mpData, 1);
      return *this;
    }


#else
      #error Routines in this section are not implemented yet without BLAS
#endif

} // namespace TNet


#endif // TNet_Vector_tcc
