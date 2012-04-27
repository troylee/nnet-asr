/** 
 * @file Matrix.cc 
 * 
 * Implementation of specialized Matrix template methods 
 */


#include "Matrix.h"

#if defined(HAVE_CLAPACK)
#include "CLAPACK-3.1.1.1/INCLUDE/f2c.h"
extern "C" {
#include "CLAPACK-3.1.1.1/INCLUDE/clapack.h"
}
// These are some stupid clapack things that we want to get rid of
#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

#endif




namespace TNet
{
  //***************************************************************************
  //***************************************************************************
#ifdef HAVE_ATLAS
  //***************************************************************************
  //***************************************************************************
  template<>
    Matrix<float> &
    Matrix<float>::
     Invert(float *LogDet, float *DetSign, bool inverse_needed)
  { 
      assert(Rows() == Cols());
      
#if defined(HAVE_CLAPACK)
      integer* pivot = new integer[mMRows];
      integer  M = Rows();
      integer  N = Cols();
      integer  LDA = mStride;
      integer  result;
      integer  l_work = std::max<integer>(1, N);
      float*   p_work = new float[l_work];

      sgetrf_(&M, &N, mpData, &LDA, pivot, &result);
	  const int pivot_offset=1;
#else
      int* pivot = new int[mMRows];
      int result = clapack_sgetrf(CblasColMajor, Rows(), Cols(), mpData, mStride, pivot);
	  const int pivot_offset=0;
#endif
      assert(result >= 0 && "Call to CLAPACK sgetrf_ or ATLAS clapack_sgetrf called with wrong arguments");
      if(result != 0) {
        Error("Matrix is singular");
      }
	  if(DetSign!=NULL){ *DetSign=1.0; for(size_t i=0;i<mMRows;i++) if(pivot[i]!=(int)i+pivot_offset) *DetSign *= -1.0; }
	  if(LogDet!=NULL||DetSign!=NULL){ // Compute log determinant...
		assert(mMRows==mMCols); // Can't take determinant of non-square matrix.
		*LogDet = 0.0;  float prod = 1.0;
		for(size_t i=0;i<mMRows;i++){ 
		  prod *= (*this)(i,i); 
		  if(i==mMRows-1 || fabs(prod)<1.0e-10 || fabs(prod)>1.0e+10){ 
			if(LogDet!=NULL) *LogDet += log(fabs(prod)); 
			if(DetSign!=NULL) *DetSign *= (prod>0?1.0:-1.0);
			prod=1.0;
		  }
		}
	  }
#if defined(HAVE_CLAPACK)
      if(inverse_needed) sgetri_(&M, mpData, &LDA, pivot, p_work, &l_work, &result);
      delete [] pivot;
#else
      if(inverse_needed) result = clapack_sgetri(CblasColMajor, Rows(), mpData, mStride, pivot);
      delete [] pivot;
#endif
      assert(result == 0 && "Call to CLAPACK sgetri_ or ATLAS clapack_sgetri called with wrong arguments");
      return *this;
    }

  
  //***************************************************************************
  //***************************************************************************
  template<>
    Matrix<double> &
    Matrix<double>::
     Invert(double *LogDet, double *DetSign, bool inverse_needed)
    { 
      assert(Rows() == Cols());
      
#if defined(HAVE_CLAPACK)
      integer* pivot = new integer[mMRows];
      integer  M = Rows();
      integer  N = Cols();
      integer  LDA = mStride;
      integer  result;
      integer  l_work = std::max<integer>(1, N);
      double*   p_work = new double[l_work];

      dgetrf_(&M, &N, mpData, &LDA, pivot, &result);
	  const int pivot_offset=1;
#else
      int* pivot = new int[mMRows];
      int result = clapack_dgetrf(CblasColMajor, Rows(), Cols(), mpData, mStride, pivot);
	  const int pivot_offset=0;
#endif
      assert(result >= 0 && "Call to CLAPACK dgetrf_ or ATLAS clapack_dgetrf called with wrong arguments");
      if(result != 0) {
        Error("Matrix is singular");
      }
	  if(DetSign!=NULL){ *DetSign=1.0; for(size_t i=0;i<mMRows;i++) if(pivot[i]!=(int)i+pivot_offset) *DetSign *= -1.0; }
	  if(LogDet!=NULL||DetSign!=NULL){ // Compute log determinant...
		assert(mMRows==mMCols); // Can't take determinant of non-square matrix.
		*LogDet = 0.0;  double prod = 1.0;
		for(size_t i=0;i<mMRows;i++){ 
		  prod *= (*this)(i,i); 
		  if(i==mMRows-1 || fabs(prod)<1.0e-10 || fabs(prod)>1.0e+10){ 
			if(LogDet!=NULL) *LogDet += log(fabs(prod)); 
			if(DetSign!=NULL) *DetSign *= (prod>0?1.0:-1.0);
			prod=1.0;
		  }
		}
	  }
#if defined(HAVE_CLAPACK)
      if(inverse_needed) dgetri_(&M, mpData, &LDA, pivot, p_work, &l_work, &result);
      delete [] pivot;
#else
      if(inverse_needed) result = clapack_dgetri(CblasColMajor, Rows(), mpData, mStride, pivot);
      delete [] pivot;
#endif
      assert(result == 0 && "Call to CLAPACK dgetri_ or ATLAS clapack_dgetri called with wrong arguments");
      return *this;
    }

  template<>
    Matrix<float> &
    Matrix<float>::
    BlasGer(const float alpha, const Vector<float>& rA, const Vector<float>& rB)
    {
      assert(rA.Dim() == mMRows && rB.Dim() == mMCols);
      cblas_sger(CblasRowMajor, rA.Dim(), rB.Dim(), alpha, rA.pData(), 1, rB.pData(), 1, mpData, mStride);
      return *this;
    }

  template<>
    Matrix<double> &
    Matrix<double>::
  BlasGer(const double alpha, const Vector<double>& rA, const Vector<double>& rB)
    {
      assert(rA.Dim() == mMRows && rB.Dim() == mMCols);
      cblas_dger(CblasRowMajor, rA.Dim(), rB.Dim(), alpha, rA.pData(), 1, rB.pData(), 1, mpData, mStride);
      return *this;
    }
  
  template<>
    Matrix<float>&
    Matrix<float>::
    BlasGemm(const float alpha,
              const Matrix<float>& rA, MatrixTrasposeType transA,
              const Matrix<float>& rB, MatrixTrasposeType transB,
              const float beta)
    {
      assert((transA == NO_TRANS && transB == NO_TRANS && rA.Cols() == rB.Rows() && rA.Rows() == Rows() && rB.Cols() == Cols())
	     || (transA ==    TRANS && transB == NO_TRANS && rA.Rows() == rB.Rows() && rA.Cols() == Rows() && rB.Cols() == Cols())
	     || (transA == NO_TRANS && transB ==    TRANS && rA.Cols() == rB.Cols() && rA.Rows() == Rows() && rB.Rows() == Cols())
	     || (transA ==    TRANS && transB ==    TRANS && rA.Rows() == rB.Cols() && rA.Cols() == Rows() && rB.Rows() == Cols()));

      cblas_sgemm(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(transA), static_cast<CBLAS_TRANSPOSE>(transB),
                  Rows(), Cols(), transA == NO_TRANS ? rA.Cols() : rA.Rows(),
                  alpha, rA.mpData, rA.mStride, rB.mpData, rB.mStride,
                  beta, mpData, mStride);
      return *this;
    }

  template<>
   Matrix<double>&
    Matrix<double>::
    BlasGemm(const double alpha,
              const Matrix<double>& rA, MatrixTrasposeType transA,
              const Matrix<double>& rB, MatrixTrasposeType transB,
              const double beta)
    {
      assert((transA == NO_TRANS && transB == NO_TRANS && rA.Cols() == rB.Rows() && rA.Rows() == Rows() && rB.Cols() == Cols())
	     || (transA ==    TRANS && transB == NO_TRANS && rA.Rows() == rB.Rows() && rA.Cols() == Rows() && rB.Cols() == Cols())
	     || (transA == NO_TRANS && transB ==    TRANS && rA.Cols() == rB.Cols() && rA.Rows() == Rows() && rB.Rows() == Cols())
	     || (transA ==    TRANS && transB ==    TRANS && rA.Rows() == rB.Cols() && rA.Cols() == Rows() && rB.Rows() == Cols()));

      cblas_dgemm(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(transA), static_cast<CBLAS_TRANSPOSE>(transB),
                  Rows(), Cols(), transA == NO_TRANS ? rA.Cols() : rA.Rows(),
                  alpha, rA.mpData, rA.mStride, rB.mpData, rB.mStride,
                  beta, mpData, mStride);
      return *this;
    }

  template<>
    Matrix<float>&
    Matrix<float>::
         Axpy(const float alpha,
              const Matrix<float>& rA, MatrixTrasposeType transA){
	int aStride = (int)rA.mStride, stride = mStride;
	float *adata=rA.mpData, *data=mpData;
	if(transA == NO_TRANS){
	  assert(rA.Rows()==Rows() && rA.Cols()==Cols());
	  for(size_t row=0;row<mMRows;row++,adata+=aStride,data+=stride)
		cblas_saxpy(mMCols, alpha, adata, 1, data, 1);
	} else {
	  assert(rA.Cols()==Rows() && rA.Rows()==Cols());
	  for(size_t row=0;row<mMRows;row++,adata++,data+=stride)
		cblas_saxpy(mMCols, alpha, adata, aStride, data, 1);
	}
	return *this;
  } 

  template<>
    Matrix<double>&
    Matrix<double>::
         Axpy(const double alpha,
              const Matrix<double>& rA, MatrixTrasposeType transA){
	int aStride = (int)rA.mStride, stride = mStride;
	double *adata=rA.mpData, *data=mpData;
	if(transA == NO_TRANS){
	  assert(rA.Rows()==Rows() && rA.Cols()==Cols());
	  for(size_t row=0;row<mMRows;row++,adata+=aStride,data+=stride)
		cblas_daxpy(mMCols, alpha, adata, 1, data, 1);
	} else {
	  assert(rA.Cols()==Rows() && rA.Rows()==Cols());
	  for(size_t row=0;row<mMRows;row++,adata++,data+=stride)
		cblas_daxpy(mMCols, alpha, adata, aStride, data, 1);
	}
	return *this;
  } 

  template <>  //non-member but friend!
  double TraceOfProduct(const Matrix<double> &A, const Matrix<double> &B){ // tr(A B), equivalent to sum of each element of A times same element in B'
	size_t aStride = A.mStride, bStride = B.mStride;
	assert(A.Rows()==B.Cols() && A.Cols()==B.Rows());
	double ans = 0.0;
	double *adata=A.mpData, *bdata=B.mpData;
	size_t arows=A.Rows(), acols=A.Cols();
	for(size_t row=0;row<arows;row++,adata+=aStride,bdata++)
	  ans += cblas_ddot(acols, adata, 1, bdata, bStride);
	return ans;
  }

  template <>  //non-member but friend!
  double TraceOfProductT(const Matrix<double> &A, const Matrix<double> &B){ // tr(A B), equivalent to sum of each element of A times same element in B'
	size_t aStride = A.mStride, bStride = B.mStride;
	assert(A.Rows()==B.Rows() && A.Cols()==B.Cols());
	double ans = 0.0;
	double *adata=A.mpData, *bdata=B.mpData;
	size_t arows=A.Rows(), acols=A.Cols();
	for(size_t row=0;row<arows;row++,adata+=aStride,bdata+=bStride)
	  ans += cblas_ddot(acols, adata, 1, bdata, 1);
	return ans;
  }


  template <>  //non-member but friend!
  float TraceOfProduct(const Matrix<float> &A, const Matrix<float> &B){ // tr(A B), equivalent to sum of each element of A times same element in B'
	size_t aStride = A.mStride, bStride = B.mStride;
	assert(A.Rows()==B.Cols() && A.Cols()==B.Rows());
	float ans = 0.0;
	float *adata=A.mpData, *bdata=B.mpData;
	size_t arows=A.Rows(), acols=A.Cols();
	for(size_t row=0;row<arows;row++,adata+=aStride,bdata++)
	  ans += cblas_sdot(acols, adata, 1, bdata, bStride);
	return ans;
  }

  template <>  //non-member but friend!
  float TraceOfProductT(const Matrix<float> &A, const Matrix<float> &B){ // tr(A B), equivalent to sum of each element of A times same element in B'
	size_t aStride = A.mStride, bStride = B.mStride;
	assert(A.Rows()==B.Rows() && A.Cols()==B.Cols());
	float ans = 0.0;
	float *adata=A.mpData, *bdata=B.mpData;
	size_t arows=A.Rows(), acols=A.Cols();
	for(size_t row=0;row<arows;row++,adata+=aStride,bdata+=bStride)
	  ans += cblas_sdot(acols, adata, 1, bdata, 1);
	return ans;
  }




#endif //HAVE_ATLAS



} //namespace STK
