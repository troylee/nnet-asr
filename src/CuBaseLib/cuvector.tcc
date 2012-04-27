
#include <cuda_runtime_api.h>

#include "Timer.h"
#include "cucommon.h"
#include "cumatrix.h"
#include "cudevice.h"

namespace TNet {

  ////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////

  template<typename _ElemT>
  CuVector<_ElemT>&
  CuVector<_ElemT>::
  Init(size_t dim)
  {
    if(mDim == dim) {
      //SetZero();
      return *this;
    }

    Destroy();

    cuSafeCall(cudaMalloc((void**)&mpCUData, dim*sizeof(_ElemT)));
    mDim = dim;
    SetZero();

    return *this;
  }

  ////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////
  
  template<typename _ElemT>
  void
  CuVector<_ElemT>::
  Destroy()
  {
    if(NULL != mpCUData) {
      cuSafeCall(cudaFree(mpCUData));
      mpCUData = NULL;
    }
    mDim = 0;
  }

  ////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////

  template<typename _ElemT>
  CuVector<_ElemT>&
  CuVector<_ElemT>::
  CopyFrom(const CuVector<_ElemT>& rSrc)
  {
    Init(rSrc.Dim());
    
    Timer tim; tim.Start();

    cuSafeCall(cudaMemcpy(mpCUData, rSrc.pCUData(), rSrc.Dim()*sizeof(_ElemT), cudaMemcpyDeviceToDevice));

    tim.End(); CuDevice::Instantiate().AccuProfile("CuVector::CopyFromD2D",tim.Val());
    return *this;
  }
  
  ////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////

  template<typename _ElemT>
  CuVector<_ElemT>&
  CuVector<_ElemT>::
  CopyFrom(const Vector<_ElemT>& rSrc)
  {
    Init(rSrc.Dim());

    Timer tim; tim.Start();

    cuSafeCall(cudaMemcpy(mpCUData, rSrc.pData(), rSrc.Dim()*sizeof(_ElemT), cudaMemcpyHostToDevice));

    tim.End(); CuDevice::Instantiate().AccuProfile("CuVector::CopyFromH2D",tim.Val());
    return *this;
  }

  ////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////

  template<typename _ElemT>
  Vector<_ElemT>&
  CuVector<_ElemT>::
  CopyTo(Vector<_ElemT>& rDst) const
  {
    if(rDst.Dim() != mDim) {
      rDst.Init(mDim);
    }

    Timer tim; tim.Start();
   
    cuSafeCall(cudaMemcpy(rDst.pData(), pCUData(), mDim*sizeof(_ElemT), cudaMemcpyDeviceToHost));

    tim.End(); CuDevice::Instantiate().AccuProfile("CuVector::CopyToD2H",tim.Val());

    return rDst;
  }

  ////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////


  template<typename _ElemT>
  void 
  CuVector<_ElemT>::
  SetZero()
  {
    Timer tim; tim.Start();
    cuSafeCall(cudaMemset(mpCUData, 0, mDim*sizeof(_ElemT)));
    tim.End(); CuDevice::Instantiate().AccuProfile("CuVector::SetZero",tim.Val());
  }


  ////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////




  ////////////////////////////////////////////////////////////////////////
  //// CuVector:: templeate specializations (float)
  ////
  template<>
  inline void CuVector<float>::SetConst(float value)
  {
    Timer tim; tim.Start();

    dim3 dimBlock(CUBLOCK);
    dim3 dimGrid(n_blocks(Dim(), CUBLOCK));
    ::MatrixDim d = { 1, Dim(), Dim() };

    cudaF_set_const(dimGrid,dimBlock,mpCUData,value,d);
    cuSafeCall(cudaGetLastError());


    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }


  template<>
  inline void CuVector<float>::AddScaled(float alpha, const CuVector<float>& vec, float beta)
  {
    Timer tim; tim.Start();

    assert(vec.Dim() == Dim());

    dim3 dimBlock(CUBLOCK);
    dim3 dimGrid(n_blocks(Dim(), CUBLOCK));
    ::MatrixDim d = { 1, Dim(), Dim() };

    cudaF_add_scaled(dimGrid,dimBlock,alpha,vec.pCUData(),beta,mpCUData,d);
    cuSafeCall(cudaGetLastError());
    
    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }


  template<>
  inline void CuVector<float>::AddColSum(float alpha, const CuMatrix<float>& mat, float beta)
  {
    Timer tim; tim.Start();

    assert(mat.Cols() == Dim());
    
    /**
     * Rows()<=512 limit due to limited shared memory
     * Cols()<=256 limit due to coalesced memory alignment:
     *             matrices with huge strides have slow access!!!
     */
    if(mat.Rows() > 512 || mat.Cols() > 256) {
      size_t dimBlock = CUBLOCK*2;
      size_t dimGrid = n_blocks(Dim(),CUBLOCK*2); 

      cudaF_add_col_sum(dimGrid,dimBlock,alpha,mat.pCUData(),beta,mpCUData,mat.Dim());
      cuSafeCall(cudaGetLastError());
    } else {
      dim3 dimBlock(mat.Rows(),1);
      dim3 dimGrid(1,Dim()); 

      cudaF_add_col_sum_reduce(dimGrid,dimBlock,alpha,mat.pCUData(),beta,mpCUData,mat.Dim());
      cuSafeCall(cudaGetLastError());
    }

    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }




  ////////////////////////////////////////////////////////////////////////
  //// CuVector:: templeate specializations (double)
  ////
  template<>
  inline void CuVector<double>::SetConst(double value)
  {
    Timer tim; tim.Start();

    dim3 dimBlock(CUBLOCK);
    dim3 dimGrid(n_blocks(Dim(), CUBLOCK));
    ::MatrixDim d = { 1, Dim(), Dim() };

    cudaD_set_const(dimGrid,dimBlock,mpCUData,value,d);
    cuSafeCall(cudaGetLastError());


    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }


  template<>
  inline void CuVector<double>::AddScaled(double alpha, const CuVector<double>& vec, double beta)
  {
    Timer tim; tim.Start();

    assert(vec.Dim() == Dim());

    dim3 dimBlock(CUBLOCK);
    dim3 dimGrid(n_blocks(Dim(), CUBLOCK));
    ::MatrixDim d = { 1, Dim(), Dim() };

    cudaD_add_scaled(dimGrid,dimBlock,alpha,vec.pCUData(),beta,mpCUData,d);
    cuSafeCall(cudaGetLastError());
    
    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }


  template<>
  inline void CuVector<double>::AddColSum(double alpha, const CuMatrix<double>& mat, double beta)
  {
    Timer tim; tim.Start();

    assert(mat.Cols() == Dim());

    size_t dimBlock = CUBLOCK*2;
    size_t dimGrid = n_blocks(Dim(),CUBLOCK*2); 

    cudaD_add_col_sum(dimGrid,dimBlock,alpha,mat.pCUData(),beta,mpCUData,mat.Dim());
    cuSafeCall(cudaGetLastError());


    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }

}



