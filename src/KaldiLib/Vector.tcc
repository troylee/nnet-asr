/** @file Vector.tcc
 *  This is an internal header file, included by other library headers.
 *  You should not attempt to use it directly.
 */

#ifndef TNet_Vector_tcc
#define TNet_Vector_tcc

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
#include "MathAux.h"
#include "Matrix.h"

namespace TNet
{
  //******************************************************************************
  //******************************************************************************
  template<typename _ElemT>
    inline Vector<_ElemT>&
    Vector<_ElemT>::
    Init(const size_t length, bool clear)
    {
	  if(mpData != NULL) Destroy();
	  if(length==0){
		mpData=NULL;
#ifdef STK_MEMALIGN_MANUAL
		mpFreeData=NULL;
#endif
		mDim=0;
		return *this;
	  }
      size_t size;
      void*  data;
      void*  free_data;

      size = align<16>(length * sizeof(_ElemT));

      if (NULL != (data = stk_memalign(16, size, &free_data))) {
        mpData        = static_cast<_ElemT*> (data);
#ifdef STK_MEMALIGN_MANUAL
        mpFreeData    = static_cast<_ElemT*> (free_data);
#endif
        mDim = length;
      } else {
        throw std::bad_alloc();
      }
      if(clear) Zero();
      return *this;
    }


  //******************************************************************************
  //******************************************************************************
  /// Copy data from another vector
  template<typename _ElemT>
    inline Vector<_ElemT>&
    Vector<_ElemT>::
    Copy(const Vector<_ElemT>& rV) {
      assert(Dim() == rV.Dim());
      Copy(rV.mpData);
      return *this;
    }

  /// Load data into the vector
  template<typename _ElemT>
    Vector<_ElemT>&
    Vector<_ElemT>::
    Copy(const _ElemT* ppData) {
      std::memcpy(this->mpData, ppData, Dim() * sizeof(_ElemT));
      return *this;
    }

  template<typename _ElemT>
  template<typename _ElemU>
    Vector<_ElemT>&
    Vector<_ElemT>::
    Copy(const Vector<_ElemU> &other){
      assert(Dim()==other.Dim());
      size_t D=Dim();
      for(size_t d=0;d<D;d++) (*this)(d) = (_ElemT) other[d];
      return *this;
  }


  //******************************************************************************
  //******************************************************************************
  template<typename _ElemT>
  Vector<_ElemT>&
  Vector<_ElemT>::
  CopyVectorizedMatrixRows(const Matrix<_ElemT> &rM) {
//    TraceLog("Dim = "+to_string(Dim())+", Rows = "+to_string(rM.Rows())+", Cols = "+to_string(rM.Cols()));
    assert(Dim() == rM.Cols()*rM.Rows());
    size_t nCols = rM.Cols();
    for(size_t r=0; r<rM.Rows(); r++)
      Range(r*nCols, nCols).Copy(rM[r]);
    return *this;
  }


  //****************************************************************************
  //****************************************************************************
  // Remove element from the vector. The vector is non reallocated
  template<typename _ElemT>
    Vector<_ElemT>&
    Vector<_ElemT>::
    RemoveElement(size_t i) {
      assert(i < mDim && "Access out of vector");
      for(size_t j = i + 1; j < mDim; j++)
        this->mpData[j - 1] = this->mpData[j];
      mDim--;
      return *this;
    }

  //****************************************************************************
  //****************************************************************************
  // The destructor
  template<typename _ElemT>
    inline void
    Vector<_ElemT>::
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
      mDim = 0;
    }


  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    inline void
    Vector<_ElemT>::
    Zero()
    {
      std::memset(mpData, 0, mDim * sizeof(_ElemT));
    }

  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    inline void
    Vector<_ElemT>::
    Set(_ElemT f)
    {
      for(size_t i=0;i<mDim;i++) mpData[i] = f;
    }


  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    Vector<_ElemT>&
    Vector<_ElemT>::
    MatrixRowStack(const Matrix<_ElemT>& rMa)
    {
      assert(mDim == rMa.Cols() * rMa.Rows());

      _ElemT*       inc_data = mpData;
      const size_t  cols     = rMa.Cols();

      for (size_t i = 0; i < rMa.Rows(); i++)
      {
        // copy the data to the propper position
        memcpy(inc_data, rMa[i], cols * sizeof(_ElemT));

        // set new copy position
        inc_data += cols;
      }
    }


  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    Vector<_ElemT>&
    Vector<_ElemT>::
	  Row(const Matrix<_ElemT> &rMa, size_t row)
    {
	  assert(row < rMa.Rows());
      const _ElemT *mRow = rMa.pRowData(row);
      // if(mDim != rMa.Cols()) Init(rMa.Cols()); // automatically resize.
      memcpy(mpData, mRow, sizeof(_ElemT)*mDim);
	  return *this;
    }

  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    Vector<_ElemT>&
    Vector<_ElemT>::
  Power(_ElemT power) // takes elements to a power.  Throws exception if could not.
    {
      for(size_t i=0;i<Dim();i++){
        _ElemT tmp = (*this)(i);
        (*this)(i) = pow(tmp, power);
        if((*this)(i) == HUGE_VAL)
          throw std::runtime_error((std::string)"Error in Vector::power, could not take " +to_string(tmp)+ " to power " +to_string((*this)(i)));
      }
      return (*this);
    }


  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    _ElemT
    Vector<_ElemT>::
    Max() const 
    {
      if(Dim()==0) throw std::runtime_error("Error in Vector::Max(), empty vector\n");
      _ElemT ans = (*this)(0);
      for(size_t i=1;i<Dim();i++) ans = std::max(ans, (*this)(i));
      return ans;
    }

  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    _ElemT
    Vector<_ElemT>::
    Min() const 
    {
      if(Dim()==0) throw std::runtime_error("Error in Vector::Min(), empty vector\n");
      _ElemT ans = (*this)(0);
      for(size_t i=1;i<Dim();i++) ans = std::min(ans, (*this)(i));
      return ans;
    }



  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    Vector<_ElemT>&
    Vector<_ElemT>::
	  Col(const Matrix<_ElemT> &rMa, size_t col)
  {
	  assert(col < rMa.Cols());
      // if(mDim != rMa.Cols()) Init(rMa.Cols()); // automatically resize.
	  for(size_t i=0;i<mDim;i++)
		mpData[i] = rMa(i,col); // can't do this efficiently so don't really bother.
	  return *this;
    }

  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    _ElemT
    Vector<_ElemT>::
    Sum() const
    {
      //note the double accumulator
      double sum = 0.0;

      for (size_t i = 0; i < mDim; ++i) {
        sum += mpData[i];
      }
      return (_ElemT)sum;
    }


  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    Vector<_ElemT>&
    Vector<_ElemT>::
    AddColSum(const Matrix<_ElemT>& rM)
    {
      // note the double accumulator
      double sum;

      assert(mDim == rM.Cols());

      for (size_t i = 0; i < mDim; ++i) {
        sum = 0.0;
        for (size_t j = 0; j < rM.Rows(); ++j) {
          sum += rM[j][i];
        }
        mpData[i] += sum;
      }
      return *this;
    }


  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    Vector<_ElemT>&
    Vector<_ElemT>::
    AddRowSum(const Matrix<_ElemT>& rM)
    {
      // note the double accumulator
      double sum;

      assert(mDim == rM.Rows());

      for (size_t i = 0; i < mDim; ++i) {
        sum = 0.0;
        for (size_t j = 0; j < rM.Cols(); ++j) {
          sum += rM[i][j];
        }
        mpData[i] += sum;
      }
      return *this;
    }


  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    _ElemT
    Vector<_ElemT>::
    LogSumExp() const
    {
      double sum = LOG_0;

      for (size_t i = 0; i < mDim; ++i) {
        sum = LogAdd(sum, mpData[i]);
      }
      return sum;
    }

  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    Vector<_ElemT>&
    Vector<_ElemT>::
    Invert() {
      for (size_t i = 0; i < mDim; ++i) {
        mpData[i] = static_cast<_ElemT>(1 / mpData[i]);
      }
      return *this;
    }

  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    Vector<_ElemT>&
    Vector<_ElemT>::
    ApplyLog() {
      for (size_t i = 0; i < mDim; ++i) {
        mpData[i] = _LOG(mpData[i]);
      }
      return *this;
    }

  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    Vector<_ElemT>&
    Vector<_ElemT>::
    ApplyLog(const Vector<_ElemT>& rV) {
      assert(mDim==rV.Dim());
      for (size_t i = 0; i < mDim; ++i) {
        mpData[i] = log(rV[i]);
      }
      return *this;
    }
    
  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    Vector<_ElemT>&
    Vector<_ElemT>::
    ApplyExp() {
      for (size_t i = 0; i < mDim; ++i) {
        mpData[i] = _EXP(mpData[i]);
      }
      return *this;
    }

  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    Vector<_ElemT>&
    Vector<_ElemT>::
    ApplySoftMax() {
      _ElemT lse = LogSumExp();

      for (size_t i = 0; i < mDim; ++i) {
        mpData[i] = exp(mpData[i] - lse);
      }
      return *this;
    }


  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    Vector<_ElemT>&
    Vector<_ElemT>::
    Add(_ElemT c)
    {
      for(size_t i = 0; i < mDim; i++) {
        mpData[i] += c;
      }
      return *this;
    }


  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    Vector<_ElemT>&
    Vector<_ElemT>::
    Subtract(_ElemT c)
    {
      for(size_t i = 0; i < mDim; i++) {
        mpData[i] -= c;
      }
      return *this;
    }


  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    Vector<_ElemT>&
    Vector<_ElemT>::
    Scale(_ElemT c)
    {
      for(size_t i = 0; i < mDim; i++) {
        mpData[i] *= c;
      }
      return *this;
    }
  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    Vector<_ElemT>&
    Vector<_ElemT>::
    MultiplyElements(const Vector<_ElemT>& rV)
    {
      assert(mDim == rV.Dim());
      for(size_t i = 0; i < mDim; i++) {
        mpData[i] *= rV[i];
      }
      return *this;
    }

  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    Vector<_ElemT>&
    Vector<_ElemT>::
    MultiplyElements(_ElemT alpha, const Vector<_ElemT>& rV, const Vector<_ElemT>& rR, _ElemT beta)
    {
      assert((mDim == rV.Dim() && mDim == rR.Dim()));
      for(size_t i = 0; i < mDim; i++) {
        mpData[i] = alpha * rV[i] * rR[i] + beta * mpData[i];
      }
      return *this;
    }

  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    Vector<_ElemT>&
    Vector<_ElemT>::
    DivideElements(const Vector<_ElemT>& rV)
    {
      assert(mDim == rV.Dim());
      for(size_t i = 0; i < mDim; i++) {
        mpData[i] /= rV[i];
      }
      return *this;
    }

  //****************************************************************************
  //****************************************************************************

  template<typename _ElemT>
    Vector<_ElemT>&
    Vector<_ElemT>::
    DivideElements(_ElemT alpha, const Vector<_ElemT>& rV, const Vector<_ElemT>& rR, _ElemT beta)
    {
      assert((mDim == rV.Dim() && mDim == rR.Dim()));
      for(size_t i = 0; i < mDim; i++) {
        mpData[i] = alpha * rV[i]/rR[i] + beta * mpData[i] ;
      }
      return *this;
    }

  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
  void Load(std::istream& rIn, Vector<_ElemT>& rV)
    { 
      std::streamoff pos = rIn.tellg();
      if(MatrixVectorIostreamControl::Flags(rIn, ACCUMULATE_INPUT)) {
        for (size_t i = 0; i < rV.Dim(); i++) {
          _ElemT tmp;
          rIn >> tmp;
          rV[i] += tmp;
        }
      } else {
        for (size_t i = 0; i < rV.Dim(); i++) {
          rIn >> rV[i];
        }
      }
      if(rIn.fail()) { 
        throw std::runtime_error("Failed to read vector from stream.  File position is "+to_string(pos));
      }
    }

  template<typename _ElemT>
    std::istream &
     operator >> (std::istream& rIn, Vector<_ElemT>& rV)
    {
      rIn >> std::ws;
      if(rIn.peek() == 'v'){ // "new" format: v <dim> 1.0 0.2 4.3 ...
        rIn.get();
        long long int tmp=-1; 
        rIn >> tmp; 
        if(rIn.fail() || tmp<0) { 
          throw std::runtime_error("Failed to read vector from stream: no size"); 
        }
        size_t tmp2 = size_t(tmp);
        assert((long long int)tmp2 == tmp);

        if(rV.Dim() != tmp2) rV.Init(tmp2);
      }
      Load(rIn,rV);
      return rIn;
    }


  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    void Save (std::ostream& rOut, const Vector<_ElemT>& rV)
    {

      for (size_t i = 0; i < rV.Dim(); i++) {
        rOut << rV[i] << ' ';
      }
      if(rOut.fail()) { 
        throw std::runtime_error("Failed to write vector to stream"); 
      }
    }


  //****************************************************************************
  //****************************************************************************
  template<typename _ElemT>
    std::ostream &
    operator << (std::ostream& rOut, const Vector<_ElemT>& rV)
    {
      rOut << "v " << rV.Dim() << "  ";
      Save(rOut,rV);
      return rOut;
    }



  //****************************************************************************
  //****************************************************************************

#ifdef HAVE_ATLAS
  template<>
    float
   BlasDot<>(const Vector<float>& rA, const Vector<float>& rB);

  template<>
   double
   BlasDot<>(const Vector<double>& rA, const Vector<double>& rB);

  template<typename _ElemT>
    inline Vector<_ElemT>&
    Vector<_ElemT>::
   DotMul(const Vector<_ElemT> &rV){
	 assert(mDim == rV.mDim);
	 const _ElemT *other_data = rV.pData();
	 _ElemT *my_data = mpData, *my_data_end = my_data+mDim;
	 for(;my_data<my_data_end;) *(my_data++) *= *(other_data++);
	 return *this;
  }

  template<>
    Vector<float>&
    Vector<float>::
    BlasAxpy(const float alpha, const Vector<float>& rV);


  template<>
    Vector<double>&
    Vector<double>::
   BlasAxpy(const double alpha, const Vector<double>& rV);


  template<>
    Vector<float>&
    Vector<float>::
    BlasGemv(const float alpha, const Matrix<float>& rM, MatrixTrasposeType trans, const Vector<float>& rV, const float beta);

  template<>
    Vector<double>&
    Vector<double>::
    BlasGemv(const double alpha, const Matrix<double>& rM, MatrixTrasposeType trans, const Vector<double>& rV, const double beta);

#else
      #error Routines in this section are not implemented yet without BLAS
#endif


  template<class _ElemT>
  _ElemT
  InnerProduct(const Vector<_ElemT> &v1, const Matrix<_ElemT> &M, const Vector<_ElemT> &v2){
    assert(v1.size()==M.Rows() && v2.size()==M.Cols());
    Vector<_ElemT> vtmp(M.Rows());
    vtmp.BlasGemv(1.0, M, NO_TRANS, v2, 0.0);
    return BlasDot(v1, vtmp);
  }


} // namespace TNet


#endif // TNet_Vector_tcc
