
#include <cuda_runtime_api.h>
#include <cublas.h>

#include "Timer.h"
#include "cucommon.h"
#include "cuvector.h"
#include "cudevice.h"

namespace TNet {

  ////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////

  template<typename _ElemT>
  CuMatrix<_ElemT>&
  CuMatrix<_ElemT>::
  Init(size_t rows, size_t cols)
  {
    if(mRows == rows && mCols == cols) {
      //SetZero();
      return *this;
    }

    Destroy();

    size_t row_bytes = cols * sizeof(_ElemT);
    size_t pitch;
    cuSafeCall(cudaMallocPitch((void**)&mpCUData, &pitch, row_bytes, rows));
    mRows = rows; mCols = cols; 
    mStride = pitch/sizeof(_ElemT);
    SetZero();
    
    return *this;
  }

  ////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////
  
  template<typename _ElemT>
  void
  CuMatrix<_ElemT>::
  Destroy()
  {
    if(NULL != mpCUData) {
      cuSafeCall(cudaFree(mpCUData));
      mpCUData = NULL;
    }
    mRows = mCols = mStride = 0;
  }

  ////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////

  template<typename _ElemT>
  CuMatrix<_ElemT>&
  CuMatrix<_ElemT>::
  CopyFrom(const CuMatrix<_ElemT>& rSrc)
  {
    Init(rSrc.Rows(),rSrc.Cols());
    
    Timer tim; tim.Start();

    size_t dst_pitch = mStride*sizeof(_ElemT);
    size_t src_pitch = rSrc.Stride()*sizeof(_ElemT);
    size_t width = rSrc.Cols()*sizeof(_ElemT);
    cuSafeCall(cudaMemcpy2D(mpCUData, dst_pitch, rSrc.pCUData(), src_pitch, width, rSrc.Rows(), cudaMemcpyDeviceToDevice));

    tim.End(); CuDevice::Instantiate().AccuProfile("CuMatrix::CopyFromD2D",tim.Val());
    return *this;
  }
  
  ////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////

  template<typename _ElemT>
  CuMatrix<_ElemT>&
  CuMatrix<_ElemT>::
  CopyFrom(const Matrix<_ElemT>& rSrc)
  {
    Init(rSrc.Rows(),rSrc.Cols());

    Timer tim; tim.Start();

    size_t dst_pitch = mStride*sizeof(_ElemT);
    size_t src_pitch = rSrc.Stride()*sizeof(_ElemT);
    size_t width = rSrc.Cols()*sizeof(_ElemT);
    cuSafeCall(cudaMemcpy2D(mpCUData, dst_pitch, rSrc.pData(), src_pitch, width, rSrc.Rows(), cudaMemcpyHostToDevice));

    tim.End(); CuDevice::Instantiate().AccuProfile("CuMatrix::CopyFromH2D",tim.Val());
    return *this;
  }

  ////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////

  template<typename _ElemT>
  Matrix<_ElemT>&
  CuMatrix<_ElemT>::
  CopyTo(Matrix<_ElemT>& rDst) const
  {
    if(rDst.Rows() != Rows()  ||  rDst.Cols() != Cols()) {
      rDst.Init(Rows(),Cols());
    }

    Timer tim; tim.Start();
   
    size_t src_pitch = mStride*sizeof(_ElemT);
    size_t dst_pitch = rDst.Stride()*sizeof(_ElemT);
    size_t width = Cols()*sizeof(_ElemT);
    cuSafeCall(cudaMemcpy2D(rDst.pData(), dst_pitch, pCUData(), src_pitch, width, Rows(), cudaMemcpyDeviceToHost));

    tim.End(); CuDevice::Instantiate().AccuProfile("CuMatrix::CopyToD2H",tim.Val());

    return rDst;
  }

  ////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////
  
  template<typename _ElemT>
  void
  CuMatrix<_ElemT>::
  CopyRows(size_t rowCnt, size_t srcOri, const CuMatrix<_ElemT>& rSrc, size_t dstOri)
  {
    assert(rowCnt+srcOri <= rSrc.Rows());
    assert(rowCnt+dstOri <= Rows());
    assert(Cols() == rSrc.Cols());
 
    Timer tim; tim.Start();

    size_t dst_pitch = mStride*sizeof(_ElemT);
    size_t src_pitch = rSrc.Stride()*sizeof(_ElemT);
    size_t width = rSrc.Cols()*sizeof(_ElemT);

    const _ElemT* p_src = rSrc.pCUData() + srcOri*rSrc.Stride();  
    _ElemT* p_dst = mpCUData + dstOri*mStride;

    cuSafeCall(cudaMemcpy2D(p_dst, dst_pitch, p_src, src_pitch, width, rowCnt, cudaMemcpyDeviceToDevice));

    tim.End(); CuDevice::Instantiate().AccuProfile("CuMatrix::CopyRowsD2D",tim.Val());
   
  }

  ////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////
  
  template<typename _ElemT>
  void
  CuMatrix<_ElemT>::
  CopyCols(size_t colCnt, size_t srcOri, const CuMatrix<_ElemT>& rSrc, size_t dstOri)
  {
    assert(colCnt+srcOri <= rSrc.Cols());
    assert(colCnt+dstOri <= Cols());
    assert(Rows() == rSrc.Rows());
 
    Timer tim; tim.Start();

    size_t dst_pitch = mStride*sizeof(_ElemT);
    size_t src_pitch = rSrc.Stride()*sizeof(_ElemT);
    size_t width = colCnt*sizeof(_ElemT);

    const _ElemT* p_src = rSrc.pCUData() + srcOri;  
    _ElemT* p_dst = mpCUData + dstOri;

    cuSafeCall(cudaMemcpy2D(p_dst, dst_pitch, p_src, src_pitch, width, Rows(), cudaMemcpyDeviceToDevice));

    tim.End(); CuDevice::Instantiate().AccuProfile("CuMatrix::CopyColsD2D",tim.Val());
   
  }
 
  ////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////
 
  template<typename _ElemT>
  void
  CuMatrix<_ElemT>::
  SetZero() 
  {
    Timer tim; tim.Start();
    cuSafeCall(cudaMemset(mpCUData, 0, mRows*mStride*sizeof(_ElemT)));
    tim.End(); CuDevice::Instantiate().AccuProfile("CuMatrix::SetZero",tim.Val());
  }


  ////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////
 
 
  ////////////////////////////////////////////////////////////////////////
  //// CuMatrix:: templeate specializations (float)
  ////
  template<> 
  inline void CuMatrix<float>::SetConst(float value)
  { 
    Timer tim; tim.Start();

    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Cols(), CUBLOCK), n_blocks(Rows(),CUBLOCK));

    cudaF_set_const(dimGrid,dimBlock,mpCUData,value,Dim());
    cuSafeCall(cudaGetLastError());

    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }


  template<> 
  inline void CuMatrix<float>::ApplyLog()
  { 
    Timer tim; tim.Start();

    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Cols(), CUBLOCK), n_blocks(Rows(),CUBLOCK));

    cudaF_apply_log(dimGrid,dimBlock,mpCUData,Dim());
    cuSafeCall(cudaGetLastError());

    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }
  
  
  template<> 
  inline void CuMatrix<float>::ApplyMask(const CuMatrix<BaseFloat>& mask)
  { 
    Timer tim; tim.Start();

    assert(mask.Rows() == Rows());
    assert(mask.Cols() == Cols());

    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Cols(), CUBLOCK), n_blocks(Rows(),CUBLOCK));

    cudaF_apply_mask(dimGrid,dimBlock,mpCUData,mask.pCUData(),Dim(),mask.Dim());
    cuSafeCall(cudaGetLastError());

    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }
  

  template<> 
  inline void CuMatrix<float>::ApplyL1(float l1)
  { 
    Timer tim; tim.Start();

    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Cols(), CUBLOCK), n_blocks(Rows(),CUBLOCK));

    cudaF_apply_l1(dimGrid,dimBlock,mpCUData,l1,Dim());
    cuSafeCall(cudaGetLastError());

    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }


  template<>
  inline void CuMatrix<float>::ScaleCols(const CuVector<float>& scale)
  {
    Timer tim; tim.Start();

    assert(scale.Dim() == Cols());

    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Cols(), CUBLOCK), n_blocks(Rows(),CUBLOCK));

    cudaF_scale_cols(dimGrid,dimBlock,mpCUData,scale.pCUData(),Dim());
    cuSafeCall(cudaGetLastError());


    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }


  
  template<>
  inline void CuMatrix<float>::ScaleRows(const CuVector<float>& scale)
  { 
    Timer tim; tim.Start();

    assert(scale.Dim() == Rows());

    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Cols(), CUBLOCK), n_blocks(Rows(),CUBLOCK));

    cudaF_scale_rows(dimGrid,dimBlock,mpCUData,scale.pCUData(),Dim());
    cuSafeCall(cudaGetLastError());


    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }



  template<>
  inline void CuMatrix<float>::AddScaled(float alpha, const CuMatrix<float>& A, float beta)
  {
    Timer tim; tim.Start();

    assert(A.Rows() == Rows());
    assert(A.Cols() == Cols());

    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Cols(), CUBLOCK), n_blocks(Rows(),CUBLOCK));

    cudaF_add_scaled(dimGrid,dimBlock,alpha,A.pCUData(),beta,mpCUData,Dim());
    cuSafeCall(cudaGetLastError());

    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }



  template<>
  inline void CuMatrix<float>::AddScaledRow(float alpha, const CuVector<float>& row, float beta)
  { 
    Timer tim; tim.Start();

    if(row.Dim() != Cols()) {
      std::ostringstream os;
      os << "Non matching dimensions: Cols:" << Cols() << " VectorDim:" << row.Dim();
      Error(os.str());
    }
    assert(row.Dim() == Cols());
   
    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Cols(), CUBLOCK), n_blocks(Rows(),CUBLOCK));

    cudaF_add_scaled_row(dimGrid,dimBlock,alpha,row.pCUData(),beta,mpCUData,Dim());
    cuSafeCall(cudaGetLastError());
    
    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }



  template<>
  inline void CuMatrix<float>::Gemm(char transa, char transb, 
            float alpha, 
            const CuMatrix<float>& A, const CuMatrix<float>& B, 
            float beta)
  { 
    // CUBLAS is col major, TNet is row major
    // keep trans..., just swap A&B argumets: A->B B->A
    size_t m = ((transb=='T' || transb=='t')? B.Rows() : B.Cols()); 
    size_t n = ((transa=='T' || transa=='t')? A.Cols() : A.Rows());
    size_t k = ((transb=='T' || transb=='t')? B.Cols() : B.Rows());
    size_t k1 = ((transa=='T' || transa=='t')? A.Rows() : A.Cols());

    assert(m == Cols());
    assert(n == Rows());
    assert(k == k1);

    #if 0
     //DEBUG MESSAGE
    std::cout << "\n" << transb << " " << transa << " " << m << " " << n << " " << k << " " <<
                alpha << " " << B << " " << B.Stride() << " " <<
                A << " " << A.Stride() << " " << beta << " " << C << " " << 
                C.Stride() << "\n" << std::flush;
    #endif

    Timer tim; tim.Start();

    cublasSgemm(transb, transa, m, n, k, 
                alpha, B.pCUData(), B.Stride(), A.pCUData(), A.Stride(), 
                beta, mpCUData, Stride());

    cuSafeCall(cublasGetError());

    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }


  template<>
  inline void CuMatrix<float>::BlasGer(float alpha, 
            const CuVector<float>& x, const CuVector<float>& y)
  { 
    // CUBLAS is col major, TNet is row major
    // just swap x and y
    assert(x.Dim() == Rows());
    assert(y.Dim() == Cols());

    Timer tim; tim.Start();
    
    cublasSger(Cols(),Rows(),alpha,y.pCUData(),1,x.pCUData(),1,mpCUData,Stride());
    cuSafeCall(cublasGetError());

    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }



  template<>
  inline void CuMatrix<float>::MulElem(const CuMatrix<float>& A)
  {
    Timer tim; tim.Start();

    assert(mCols == A.Cols());
    assert(mRows == A.Rows());
    assert(mStride == A.Stride());
    
    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Cols(), CUBLOCK), n_blocks(Rows(),CUBLOCK));

    cudaF_mul_elem(dimGrid,dimBlock,mpCUData, A.pCUData(), Dim());
    cuSafeCall(cudaGetLastError());
    
    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }


  template<>
  inline void CuMatrix<float>::LogElem()
  {
    Timer tim; tim.Start();

    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Cols(), CUBLOCK), n_blocks(Rows(),CUBLOCK));

    cudaF_log_elem(dimGrid,dimBlock,mpCUData, Dim());
    cuSafeCall(cudaGetLastError());
    
    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }





  ////////////////////////////////////////////////////////////////////////
  //// CuMatrix:: templeate specializations (double)
  ////
  template<> 
  inline void CuMatrix<double>::SetConst(double value)
  { 
    Timer tim; tim.Start();

    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Cols(), CUBLOCK), n_blocks(Rows(),CUBLOCK));

    cudaD_set_const(dimGrid,dimBlock,mpCUData,value,Dim());
    cuSafeCall(cudaGetLastError());

    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }


  template<> 
  inline void CuMatrix<double>::ApplyLog()
  { 
    Timer tim; tim.Start();

    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Cols(), CUBLOCK), n_blocks(Rows(),CUBLOCK));

    cudaD_apply_log(dimGrid,dimBlock,mpCUData,Dim());
    cuSafeCall(cudaGetLastError());

    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }


  template<>
  inline void CuMatrix<double>::ScaleCols(const CuVector<double>& scale)
  {
    Timer tim; tim.Start();

    assert(scale.Dim() == Cols());

    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Cols(), CUBLOCK), n_blocks(Rows(),CUBLOCK));

    cudaD_scale_cols(dimGrid,dimBlock,mpCUData,scale.pCUData(),Dim());
    cuSafeCall(cudaGetLastError());


    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }


  
  template<>
  inline void CuMatrix<double>::ScaleRows(const CuVector<double>& scale)
  { 
    Timer tim; tim.Start();

    assert(scale.Dim() == Rows());

    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Cols(), CUBLOCK), n_blocks(Rows(),CUBLOCK));

    cudaD_scale_rows(dimGrid,dimBlock,mpCUData,scale.pCUData(),Dim());
    cuSafeCall(cudaGetLastError());


    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }



  template<>
  inline void CuMatrix<double>::AddScaled(double alpha, const CuMatrix<double>& A, double beta)
  {
    Timer tim; tim.Start();

    assert(A.Rows() == Rows());
    assert(A.Cols() == Cols());

    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Cols(), CUBLOCK), n_blocks(Rows(),CUBLOCK));

    cudaD_add_scaled(dimGrid,dimBlock,alpha,A.pCUData(),beta,mpCUData,Dim());
    cuSafeCall(cudaGetLastError());

    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }



  template<>
  inline void CuMatrix<double>::AddScaledRow(double alpha, const CuVector<double>& row, double beta)
  { 
    Timer tim; tim.Start();

    assert(row.Dim() == Cols());
   
    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Cols(), CUBLOCK), n_blocks(Rows(),CUBLOCK));

    cudaD_add_scaled_row(dimGrid,dimBlock,alpha,row.pCUData(),beta,mpCUData,Dim());
    cuSafeCall(cudaGetLastError());
    
    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }



  template<>
  inline void CuMatrix<double>::Gemm(char transa, char transb, 
            double alpha, 
            const CuMatrix<double>& A, const CuMatrix<double>& B, 
            double beta)
  { 
    // CUBLAS is col major, TNet is row major
    // keep trans..., just swap A&B argumets: A->B B->A
    size_t m = ((transb=='T' || transb=='t')? B.Rows() : B.Cols()); 
    size_t n = ((transa=='T' || transa=='t')? A.Cols() : A.Rows());
    size_t k = ((transb=='T' || transb=='t')? B.Cols() : B.Rows());
    size_t k1 = ((transa=='T' || transa=='t')? A.Rows() : A.Cols());

    assert(m == Cols());
    assert(n == Rows());
    assert(k == k1);

    #if 0
     //DEBUG MESSAGE
    std::cout << "\n" << transb << " " << transa << " " << m << " " << n << " " << k << " " <<
                alpha << " " << B << " " << B.Stride() << " " <<
                A << " " << A.Stride() << " " << beta << " " << C << " " << 
                C.Stride() << "\n" << std::flush;
    #endif

    Timer tim; tim.Start();

    cublasDgemm(transb, transa, m, n, k, 
                alpha, B.pCUData(), B.Stride(), A.pCUData(), A.Stride(), 
                beta, mpCUData, Stride());

    cuSafeCall(cublasGetError());

    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }

  template<>
  inline void CuMatrix<double>::BlasGer(double alpha, 
            const CuVector<double>& x, const CuVector<double>& y)
  { 
    // CUBLAS is col major, TNet is row major
    // just swap x and y
    assert(x.Dim() == Rows());
    assert(y.Dim() == Cols());

    Timer tim; tim.Start();
    
    cublasDger(Cols(),Rows(),alpha,y.pCUData(),1,x.pCUData(),1,mpCUData,Stride());
    cuSafeCall(cublasGetError());

    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }




  template<>
  inline void CuMatrix<double>::MulElem(const CuMatrix<double>& A)
  {
    Timer tim; tim.Start();

    assert(mCols == A.Cols());
    assert(mRows == A.Rows());
    assert(mStride == A.Stride());
    
    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Cols(), CUBLOCK), n_blocks(Rows(),CUBLOCK));

    cudaD_mul_elem(dimGrid,dimBlock,mpCUData, A.pCUData(), Dim());
    cuSafeCall(cudaGetLastError());
    
    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }


  template<>
  inline void CuMatrix<double>::LogElem()
  {
    Timer tim; tim.Start();

    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Cols(), CUBLOCK), n_blocks(Rows(),CUBLOCK));

    cudaD_log_elem(dimGrid,dimBlock,mpCUData, Dim());
    cuSafeCall(cudaGetLastError());
    
    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }


}
