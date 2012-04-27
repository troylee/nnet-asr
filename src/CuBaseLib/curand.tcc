
#include <cstdlib>
#include "curandkernels.h"


namespace TNet {
 
 

  template<typename T>
  inline void
  CuRand<T>::
  SeedGpu(size_t rows, size_t cols)
  {
    Matrix<unsigned> mat(rows,cols);
    SeedRandom(mat);
    z1.CopyFrom(mat);
    SeedRandom(mat);
    z2.CopyFrom(mat);
    SeedRandom(mat);
    z3.CopyFrom(mat);
    SeedRandom(mat);
    z4.CopyFrom(mat);

    /*
    std::cout << "RANDININIT" << std::endl;
    z1.Print();
    z2.Print();
    z3.Print();
    z4.Print();
    std::cout << "RANDININIT" << std::endl;
    */

    tmp.Init(rows,cols);
  }



  template<typename T>  
  inline void 
  CuRand<T>::
  SeedRandom(Matrix<unsigned>& mat) {
    for(size_t j=0; j<mat.Rows(); j++) {
      for(size_t i=0; i<mat.Cols(); i++) {
        unsigned value = 0;
        while(value <= 128) { value = lrand48(); }
        mat(j,i) = value;
      }
    }
  }


  template<typename T>
  inline void
  CuRand<T>::
  AddGaussNoise(CuMatrix<T>& tgt, T gscale)
  { 
    GaussRand(tmp);
    tgt.AddScaled(gscale,tmp,1.0);
  }







  ////////////////////////////////////////////////////////////////////////////
  //// invalid general wrappers over CUDA kernels
  template<typename T>
  inline void
  CuRand<T>::
  Rand(CuMatrix<T>& tgt)
  { Error("Unimplemented"); }
  
  template<typename T>
  inline void
  CuRand<T>::
  GaussRand(CuMatrix<T>& tgt)
  { Error("Unimplemented"); }
 
  template<typename T>
  inline void
  CuRand<T>::
  BinarizeProbs(const CuMatrix<T>& probs, CuMatrix<T>& states)
  { Error("Unimplemented"); }


  //////////////////////////////////////////////////////////////////////////
  //// float specializations
  template<>
  inline void
  CuRand<float>::
  Rand(CuMatrix<float>& tgt)
  {
    Timer tim; tim.Start();

    tgt.Init(z1.Rows(), z1.Cols());
  
    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(tgt.Cols(), CUBLOCK), n_blocks(tgt.Rows(),CUBLOCK));

    cudaF_rand(dimGrid,dimBlock,tgt.pCUData(), z1.pCUData(), z2.pCUData(), z3.pCUData(), z4.pCUData(),tgt.Dim());

    cuSafeCall(cudaGetLastError());
    
    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }
    
  
  template<>
  inline void
  CuRand<float>::
  GaussRand(CuMatrix<float>& tgt)
  {

    Timer tim; tim.Start();

    tgt.Init(z1.Rows(), z1.Cols());
  
    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(tgt.Cols(), CUBLOCK), n_blocks(tgt.Rows(),CUBLOCK));

    cudaF_gauss_rand(dimGrid,dimBlock,tgt.pCUData(), z1.pCUData(), z2.pCUData(), z3.pCUData(), z4.pCUData(),tgt.Dim());

    cuSafeCall(cudaGetLastError());


    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }

 
  template<>
  inline void
  CuRand<float>::
  BinarizeProbs(const CuMatrix<float>& probs, CuMatrix<float>& states)
  {
    if(probs.Rows() != z1.Rows() || probs.Cols() != z1.Cols()) {
      Error("Non matching dims!!");
    }

    states.Init(z1.Rows(),z1.Cols());
    Rand(tmp);

    Timer tim; tim.Start();

    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(z1.Cols(), CUBLOCK), n_blocks(z1.Rows(),CUBLOCK));

    cudaF_binarize_probs(dimGrid,dimBlock,states.pCUData(), probs.pCUData(), tmp.pCUData(),states.Dim());

    cuSafeCall(cudaGetLastError());

    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }


  //////////////////////////////////////////////////////////////////////////
  //// double specializations
  template<>
  inline void
  CuRand<double>::
  Rand(CuMatrix<double>& tgt)
  {
    Timer tim; tim.Start();

    tgt.Init(z1.Rows(), z1.Cols());
  
    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(tgt.Cols(), CUBLOCK), n_blocks(tgt.Rows(),CUBLOCK));

    cudaD_rand(dimGrid,dimBlock,tgt.pCUData(), z1.pCUData(), z2.pCUData(), z3.pCUData(), z4.pCUData(),tgt.Dim());

    cuSafeCall(cudaGetLastError());
    
    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }
    
  
  template<>
  inline void
  CuRand<double>::
  GaussRand(CuMatrix<double>& tgt)
  {

    Timer tim; tim.Start();

    tgt.Init(z1.Rows(), z1.Cols());
  
    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(tgt.Cols(), CUBLOCK), n_blocks(tgt.Rows(),CUBLOCK));

    cudaD_gauss_rand(dimGrid,dimBlock,tgt.pCUData(), z1.pCUData(), z2.pCUData(), z3.pCUData(), z4.pCUData(),tgt.Dim());

    cuSafeCall(cudaGetLastError());


    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }

 
  template<>
  inline void
  CuRand<double>::
  BinarizeProbs(const CuMatrix<double>& probs, CuMatrix<double>& states)
  {
    if(probs.Rows() != z1.Rows() || probs.Cols() != z1.Cols()) {
      Error("Non matching dims!!");
    }

    states.Init(z1.Rows(),z1.Cols());
    Rand(tmp);

    Timer tim; tim.Start();

    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(z1.Cols(), CUBLOCK), n_blocks(z1.Rows(),CUBLOCK));

    cudaD_binarize_probs(dimGrid,dimBlock,states.pCUData(), probs.pCUData(), tmp.pCUData(),states.Dim());

    cuSafeCall(cudaGetLastError());

    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }



}
