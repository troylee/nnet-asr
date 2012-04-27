
/** @file Matrix.tcc
 *  This is an internal header file, included by other library headers.
 *  You should not attempt to use it directly.
 */


#ifndef TNet_Matrix_tcc
#define TNet_Matrix_tcc

//#pragma GCC system_header

#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <fstream>
#include <iomanip>
#include <typeinfo>
#include <algorithm>
#include <limits>
#include <vector>
#include "Common.h"

#ifndef _XOPEN_SOURCE
  #define _XOPEN_SOURCE 600
#endif


#ifdef HAVE_ATLAS
extern "C"{
  #include <cblas.h>
}
#endif


#include "Common.h"
#include "Vector.h"
namespace TNet
{

//******************************************************************************
  template<typename _ElemT>
  Matrix<_ElemT> &
  Matrix<_ElemT>::
  Init(const size_t rows,
       const size_t cols, 
       bool clear)
  {
    if(mpData != NULL) Destroy();
    if(rows*cols == 0){
      assert(rows==0 && cols==0);
      mMRows=rows; 
      mMCols=cols;
#ifdef STK_MEMALIGN_MANUAL
      mpFreeData=NULL;
#endif
      mpData=NULL;
      return *this;
    }
    // initialize some helping vars
    size_t  skip;
    size_t  real_cols;
    size_t  size;
    void*   data;       // aligned memory block
    void*   free_data;  // memory block to be really freed

    // compute the size of skip and real cols
    skip      = ((16 / sizeof(_ElemT)) - cols % (16 / sizeof(_ElemT))) % (16 / sizeof(_ElemT));
    real_cols = cols + skip;
    size      = rows * real_cols * sizeof(_ElemT);

    // allocate the memory and set the right dimensions and parameters

    if (NULL != (data = stk_memalign(16, size, &free_data)))
    {
      mpData        = static_cast<_ElemT *> (data);
#ifdef STK_MEMALIGN_MANUAL
      mpFreeData    = static_cast<_ElemT *> (free_data);
#endif
      mMRows      = rows;
      mMCols      = cols;
      mStride  = real_cols;
    }
    else
    {
      throw std::bad_alloc();
    }
    if(clear) Zero();
    return *this;
  } //

  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    template<typename _ElemU>
    Matrix<_ElemT> &
    Matrix<_ElemT>::
  Copy(const Matrix<_ElemU> & rM, MatrixTrasposeType Trans)
    {
      if(Trans==NO_TRANS){
        assert(mMRows == rM.Rows() && mMCols == rM.Cols());
        for(size_t i = 0; i < mMRows; i++) 
          (*this)[i].Copy(rM[i]);
        return *this;
      } else {
        assert(mMCols == rM.Rows() && mMRows == rM.Cols());        
        for(size_t i = 0; i < mMRows; i++) 
          for(size_t j = 0; j < mMCols; j++)
            (*this)(i,j) = rM(j,i);
        return *this;
      }
    }



  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
  Matrix<_ElemT> &
  Matrix<_ElemT>::
  CopyVectorSplicedRows(const Vector<_ElemT> &rV, const size_t nRows, const size_t nCols) {
    assert(rV.Dim() == nRows*nCols);
    mMRows = nRows;
    mMCols = nCols;

    for(size_t r=0; r<mMRows; r++)
      for(size_t c=0; c<mMCols; c++)
        (*this)(r,c) = rV(r*mMCols + c);

      return *this;
    }

  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    Matrix<_ElemT> &
    Matrix<_ElemT>::
  RemoveRow(size_t i)
  {
    assert(i < mMRows && "Access out of matrix");
    for(size_t j = i + 1; j < mMRows; j++)
      (*this)[j - 1].Copy((*this)[j]);
    mMRows--;
    return *this;
  }


  //****************************************************************************
  //****************************************************************************
  // The destructor
  template<typename _ElemT>
    void
    Matrix<_ElemT>::
    Destroy()
    {
      // we need to free the data block if it was defined
#ifndef STK_MEMALIGN_MANUAL
      if (NULL != mpData) free(mpData);
#else
      if (NULL != mpData) free(mpFreeData);
      mpFreeData = NULL;
#endif

      mpData = NULL;
      mMRows = mMCols = 0;
    }

  //****************************************************************************
  //****************************************************************************
//  template<typename _ElemT>
//  void
//  Matrix<_ElemT>::
//  VectorizeRows(Vector<_ElemT> &rV) {
//#ifdef PARANIOD
//    assert(rV.Dim() == mMRows*mMCols);
//#endif
//    for(size_t r=0; r<mMRows; r++) {
//      rV.Range((r-1)*mMCols, mMCols).Copy((*this)[r]);
//    }
//  }


  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    bool
    Matrix<_ElemT>::
    LoadHTK(const char* pFileName)
    {
      HtkHeader htk_hdr;

      FILE *fp = fopen(pFileName, "rb");
      if(!fp)
      {
        return false;
      }

      read(fileno(fp), &htk_hdr, sizeof(htk_hdr));

      swap4(htk_hdr.mNSamples);
      swap4(htk_hdr.mSamplePeriod);
      swap2(htk_hdr.mSampleSize);
      swap2(htk_hdr.mSampleKind);

      Init(htk_hdr.mNSamples, htk_hdr.mSampleSize / sizeof(float));

      size_t i;
      size_t j;
      if (typeid(_ElemT) == typeid(float))
      {
        for (i=0; i< Rows(); ++i) {
          read(fileno(fp), (*this).pRowData(i), Cols() * sizeof(float));

          for(j = 0; j < Cols(); j++) {
            swap4(((*this)(i,j)));
          }
        }
      }
      else
      {
        float *pmem = new (std::nothrow) float[Cols()];
        if (!pmem)
        {
          fclose(fp);
          return false;
        }

        for(i = 0; i < Rows(); i++) {
          read(fileno(fp), pmem, Cols() * sizeof(float));

          for (j = 0; j < Cols(); ++j) {
            swap4(pmem[j]);
            (*this)(i,j) = static_cast<_ElemT>(pmem[j]);
          }
        }
        delete [] pmem;
      }

      fclose(fp);

      return true;
    }

  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    Matrix<_ElemT> &
    Matrix<_ElemT>::
    DotMul(const ThisType& a)
    {
      size_t i;
      size_t j;

      for (i = 0; i < mMRows; ++i) {
        for (j = 0; j < mMCols; ++j) {
          (*this)(i,j) *= a(i,j);
        }
      }
      return *this;
    }

  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    _ElemT &
    Matrix<_ElemT>::
    Sum() const
    {
      double sum = 0.0;

      for (size_t i = 0; i < Rows(); ++i) {
        for (size_t j = 0; j < Cols(); ++j) {
          sum += (*this)(i,j);
        }
      }

      return sum;
    }

  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    Matrix<_ElemT>&
    Matrix<_ElemT>::
    Scale(_ElemT alpha)
    {
#if 0
      for (size_t i = 0; i < Rows(); ++i) 
        for (size_t j = 0; j < Cols(); ++j) 
          (*this)(i,j) *= alpha;
#else
      for (size_t i = 0; i < Rows(); ++i) {
        _ElemT* p_data = pRowData(i);
        for (size_t j = 0; j < Cols(); ++j) {
          *p_data++ *= alpha; 
        }
      }
#endif
      return *this;
    }

  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    Matrix<_ElemT>&
    Matrix<_ElemT>::
    ScaleRows(const Vector<_ElemT>& scale) // scales each row by scale[i].
    {
      assert(scale.Dim() == Rows());
      size_t M = Rows(), N = Cols();

      for (size_t i = 0; i < M; i++) {
        _ElemT this_scale = scale(i);
        for (size_t j = 0; j < N; j++) {
          (*this)(i,j) *= this_scale;
        }
      }
      return *this;
     }

  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    Matrix<_ElemT>&
    Matrix<_ElemT>::
    ScaleCols(const Vector<_ElemT>& scale) // scales each column by scale[i].
    {
      assert(scale.Dim() == Cols());
      for (size_t i = 0; i < Rows(); i++) {
        for (size_t j = 0; j < Cols(); j++) {
          _ElemT this_scale = scale(j);
          (*this)(i,j) *= this_scale;
        }
      }
      return *this;
    }


  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
  Matrix<_ElemT>&
  Matrix<_ElemT>::
  Add(const Matrix<_ElemT>& rMatrix) 
  {
    assert(rMatrix.Cols() == Cols());
    assert(rMatrix.Rows() == Rows());
      
#if 0
    //this can be slow
    for (size_t i = 0; i < Rows(); i++) {
      for (size_t j = 0; j < Cols(); j++) {
        (*this)(i,j) += rMatrix(i,j);
      }
    }
#else
    //this will be faster (but less secure)
    for(size_t i=0; i<Rows(); i++) {
      const _ElemT* p_src = rMatrix.pRowData(i);
      _ElemT* p_dst = pRowData(i);
      for(size_t j=0; j<Cols(); j++) {
        *p_dst++ += *p_src++;
      }
    }
#endif
    return *this;
  }



  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
  Matrix<_ElemT>&
  Matrix<_ElemT>::
  AddScaled(_ElemT alpha, const Matrix<_ElemT>& rMatrix) 
  {
    assert(rMatrix.Cols() == Cols());
    assert(rMatrix.Rows() == Rows());
      
#if 0
    //this can be slow
    for (size_t i = 0; i < Rows(); i++) {
      for (size_t j = 0; j < Cols(); j++) {
        (*this)(i,j) += rMatrix(i,j) * alpha;
      }
    }
#else
  /*
    //this will be faster (but less secure)
    for(size_t i=0; i<Rows(); i++) {
      const _ElemT* p_src = rMatrix.pRowData(i);
      _ElemT* p_dst = pRowData(i);
      for(size_t j=0; j<Cols(); j++) {
        *p_dst++ += *p_src++ * alpha;
      }
    }
    */

  //let's use BLAS
  for(size_t i=0; i<Rows(); i++) {
    (*this)[i].BlasAxpy(alpha, rMatrix[i]);
  }
#endif
    return *this;
  }



  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
  Matrix<_ElemT>&
  Matrix<_ElemT>::
  ApplyLog()
  {
      
#if 0
    //this can be slow
    for (size_t i = 0; i < Rows(); i++) {
      for (size_t j = 0; j < Cols(); j++) {
        (*this)(i,j) = += _LOG((*this)(i,j));
      }
    }
#else
    //this will be faster (but less secure)
    for(size_t i=0; i<Rows(); i++) {
      _ElemT* p_data = pRowData(i);
      for(size_t j=0; j<Cols(); j++) {
        *p_data = _LOG(*p_data);
        p_data++;
      }
    }
#endif
    return *this;
  }



  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    Matrix<_ElemT> &
    Matrix<_ElemT>::
    Zero()
    {
    for(size_t row=0;row<mMRows;row++)
    memset(mpData + row*mStride, 0, sizeof(_ElemT)*mMCols);
      return *this;
    }

  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    Matrix<_ElemT> &
    Matrix<_ElemT>::
    Unit()
    {
    for(size_t row=0;row<std::min(mMRows,mMCols);row++){
    memset(mpData + row*mStride, 0, sizeof(_ElemT)*mMCols);
    (*this)(row,row) = 1.0;
    }
      return *this;
    }


  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    void
    Matrix<_ElemT>::
    PrintOut(char* file)
    {
      FILE* f = fopen(file, "w");
      unsigned i,j;
      fprintf(f, "%dx%d\n", this->mMRows, this->mMCols);

      for(i=0; i<this->mMRows; i++)
      {
        _ElemT*   row = (*this)[i];

        for(j=0; j<this->mStride; j++){
          fprintf(f, "%20.17f ",row[j]);
        }
        fprintf(f, "\n");
      }

      fclose(f);
    }


  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    void
    Matrix<_ElemT>::
    ReadIn(char* file)
    {
      FILE* f = fopen(file, "r");
      int  i = 0;
      int j = 0;
      fscanf(f, "%dx%d\n", &i,&j);
      fprintf(stderr, "%dx%d\n", i,j);

      for(i=0; i<this->mMRows; i++)
      {
        _ElemT*   row = (*this)[i];

        for(j=0; j<this->mStride; j++){
          fscanf(f, "%f ",&row[j]);
        }
        //fprintf(f, "\n");
      }

      fclose(f);
    }


  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    void Save (std::ostream &rOut, const Matrix<_ElemT> &rM)
    {
      for (size_t i = 0; i < rM.Rows(); i++) {
        for (size_t j = 0; j < rM.Cols(); j++) {
          rOut << rM(i,j) << ' ';
        }
        rOut << '\n';
      }
      if(rOut.fail()) 
        throw std::runtime_error("Failed to write matrix to stream");
    }


  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    std::ostream &
    operator << (std::ostream & rOut, const Matrix<_ElemT> & rM)
    {
      rOut << "m " << rM.Rows() << ' ' << rM.Cols() << '\n';
      Save(rOut, rM);
      return rOut;
    }



  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    void Load (std::istream & rIn, Matrix<_ElemT> & rM)
    {
      if(MatrixVectorIostreamControl::Flags(rIn, ACCUMULATE_INPUT)) {
        for (size_t i = 0; i < rM.Rows(); i++) {
          std::streamoff pos = rIn.tellg();
          for (size_t j = 0; j < rM.Cols(); j++) {
            _ElemT tmp;
            rIn >> tmp;
            rM(i,j) += tmp;
            if(rIn.fail()){
              throw std::runtime_error("Failed to read matrix from stream.  File position is "+to_string(pos));
            }        
          }
        }
      } else {
        for (size_t i = 0; i < rM.Rows(); i++) {
          std::streamoff pos = rIn.tellg();
          for (size_t j = 0; j < rM.Cols(); j++) {
            rIn >> rM(i,j);
            if(rIn.fail()){
              throw std::runtime_error("Failed to read matrix from stream.  File position is "+to_string(pos));
            }        

          }
        }
      }
    }


  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    std::istream &
    operator >> (std::istream & rIn, Matrix<_ElemT> & rM)
    {
      while(isascii(rIn.peek()) && isspace(rIn.peek())) rIn.get(); // eat up space.
      if(rIn.peek() == 'm'){ // "new" format: m <nrows> <ncols> \n 1.0 0.2 4.3 ...
        rIn.get();// eat up the 'm'.
        long long int nrows=-1; rIn>>nrows; 
        long long int ncols=-1; rIn>>ncols; 
        if(rIn.fail()||nrows<0||ncols<0){ throw std::runtime_error("Failed to read matrix from stream: no size\n"); }

        size_t nrows2 = size_t(nrows), ncols2 = size_t(ncols);
        assert((long long int)nrows2 == nrows && (long long int)ncols2 == ncols);

        if(rM.Rows()!=nrows2 || rM.Cols()!=ncols2) rM.Init(nrows2,ncols2);
      }
      Load(rIn,rM);
      return rIn;
    }



  //****************************************************************************
  //****************************************************************************
  // Constructor
  template<typename _ElemT>
    SubMatrix<_ElemT>::
    SubMatrix(const Matrix<_ElemT>& rT, // Matrix cannot be const because SubMatrix can change its contents.  Would have to have a ConstSubMatrix or something...
              const size_t    ro,
              const size_t    r,
              const size_t    co,
              const size_t    c)
    {
      assert(ro >= 0 && ro <= rT.Rows());
      assert(co >= 0 && co <= rT.Cols());
      assert(r  >  0 && r  <= rT.Rows() - ro);
      assert(c  >  0 && c  <= rT.Cols() - co);
      // point to the begining of window
      Matrix<_ElemT>::mMRows = r;
      Matrix<_ElemT>::mMCols = c;
      Matrix<_ElemT>::mStride = rT.Stride();
      Matrix<_ElemT>::mpData = rT.pData_workaround() + co + ro * rT.Stride();
    }



#ifdef HAVE_ATLAS

  template<>
    Matrix<float> &
    Matrix<float>::
   BlasGer(const float alpha, const Vector<float>& rA, const Vector<float>& rB);


  template<>
    Matrix<double> &
    Matrix<double>::
   BlasGer(const double alpha, const Vector<double>& rA, const Vector<double>& rB);


  template<>
    Matrix<float>&
    Matrix<float>::
    BlasGemm(const float alpha,
              const Matrix<float>& rA, MatrixTrasposeType transA,
              const Matrix<float>& rB, MatrixTrasposeType transB,
       const float beta);

  template<>
   Matrix<double>&
    Matrix<double>::
    BlasGemm(const double alpha,
              const Matrix<double>& rA, MatrixTrasposeType transA,
              const Matrix<double>& rB, MatrixTrasposeType transB,
       const double beta);

  template<>
    Matrix<float>&
    Matrix<float>::
         Axpy(const float alpha,
              const Matrix<float>& rA, MatrixTrasposeType transA);

  template<>
    Matrix<double>&
    Matrix<double>::
         Axpy(const double alpha,
              const Matrix<double>& rA, MatrixTrasposeType transA);

  template <>  // non-member so automatic namespace lookup can occur.
  double TraceOfProduct(const Matrix<double> &A, const Matrix<double> &B);

  template <>  // non-member so automatic namespace lookup can occur.
  double TraceOfProductT(const Matrix<double> &A, const Matrix<double> &B);

  template <>  // non-member so automatic namespace lookup can occur.
  float TraceOfProduct(const Matrix<float> &A, const Matrix<float> &B);

  template <>  // non-member so automatic namespace lookup can occur.
  float TraceOfProductT(const Matrix<float> &A, const Matrix<float> &B);

  

#else // HAVE_ATLAS
      #error Routines in this section are not implemented yet without BLAS
#endif // HAVE_ATLAS

  template<class _ElemT>
  bool
  Matrix<_ElemT>::
  IsSymmetric(_ElemT cutoff) const {
  size_t R=Rows(), C=Cols();
  if(R!=C) return false;
  _ElemT bad_sum=0.0, good_sum=0.0;
  for(size_t i=0;i<R;i++){
    for(size_t j=0;j<i;j++){
    _ElemT a=(*this)(i,j),b=(*this)(j,i), avg=0.5*(a+b), diff=0.5*(a-b);    
    good_sum += fabs(avg); bad_sum += fabs(diff);
    }
    good_sum += fabs((*this)(i,i));
  }
  if(bad_sum > cutoff*good_sum) return false;
  return true;
  }

  template<class _ElemT>
  bool
  Matrix<_ElemT>::
  IsDiagonal(_ElemT cutoff) const{
  size_t R=Rows(), C=Cols();
  _ElemT bad_sum=0.0, good_sum=0.0;
  for(size_t i=0;i<R;i++){
    for(size_t j=0;j<C;j++){
    if(i==j) good_sum += (*this)(i,j);
    else bad_sum += (*this)(i,j);
    }
  }
  return (!(bad_sum > good_sum * cutoff));
  }

  template<class _ElemT>
  bool
  Matrix<_ElemT>::
  IsUnit(_ElemT cutoff) const {
  size_t R=Rows(), C=Cols();
  if(R!=C) return false;
  _ElemT bad_sum=0.0;
  for(size_t i=0;i<R;i++)
    for(size_t j=0;j<C;j++)
    bad_sum += fabs( (*this)(i,j) - (i==j?1.0:0.0));
  return (bad_sum <= cutoff);
  }

  template<class _ElemT>
  bool
  Matrix<_ElemT>::
  IsZero(_ElemT cutoff)const {
  size_t R=Rows(), C=Cols();
  _ElemT bad_sum=0.0;
  for(size_t i=0;i<R;i++)
    for(size_t j=0;j<C;j++)
    bad_sum += fabs( (*this)(i,j) );
  return (bad_sum <= cutoff);
  }

  template<class _ElemT>
  _ElemT
  Matrix<_ElemT>::
  FrobeniusNorm() const{
  size_t R=Rows(), C=Cols();
  _ElemT sum=0.0;
  for(size_t i=0;i<R;i++)
    for(size_t j=0;j<C;j++){
        _ElemT tmp = (*this)(i,j);
    sum +=  tmp*tmp;
      }
    return sqrt(sum);
  }

  template<class _ElemT>
  _ElemT
  Matrix<_ElemT>::
  LargestAbsElem() const{
  size_t R=Rows(), C=Cols();
  _ElemT largest=0.0;
  for(size_t i=0;i<R;i++)
    for(size_t j=0;j<C;j++)
        largest = std::max(largest, (_ElemT)fabs((*this)(i,j)));
    return largest;
  }



  // Uses SVD to compute the eigenvalue decomposition of a symmetric positive semidefinite 
  //   matrix: 
  // (*this) = rU * diag(rS) * rU^T, with rU an orthogonal matrix so rU^{-1} = rU^T.
  // Does this by computing svd (*this) = U diag(rS) V^T ... answer is just U diag(rS) U^T.
  // Throws exception if this failed to within supplied precision (typically because *this was not 
  // symmetric positive definite).  
  
  

  template<class _ElemT>
  _ElemT
  Matrix<_ElemT>::
  LogAbsDeterminant(_ElemT *DetSign){
    _ElemT LogDet;
  Matrix<_ElemT> tmp(*this);
  tmp.Invert(&LogDet, DetSign, false); // false== output not needed (saves some computation).
    return LogDet;
  }

}// namespace TNet

// #define TNet_Matrix_tcc
#endif
