#ifndef _cuda_rand_kernels_h_
#define _cuda_rand_kernels_h_


#include "cukernels.h"


extern "C" {
  //**************
  //float
  //
  void cudaF_rand(dim3 Gr, dim3 Bl, float* mat, unsigned* z1, unsigned* z2, unsigned* z3, unsigned* z4, MatrixDim d);


  void cudaF_gauss_rand(dim3 Gr, dim3 Bl, float* mat, unsigned* z1, unsigned* z2, unsigned* z3, unsigned* z4, MatrixDim d);


  void cudaF_binarize_probs(dim3 Gr, dim3 Bl, float* states, const float* probs, float* rand, MatrixDim d);
  
  //**************
  //double
  //
  void cudaD_rand(dim3 Gr, dim3 Bl, double* mat, unsigned* z1, unsigned* z2, unsigned* z3, unsigned* z4, MatrixDim d);


  void cudaD_gauss_rand(dim3 Gr, dim3 Bl, double* mat, unsigned* z1, unsigned* z2, unsigned* z3, unsigned* z4, MatrixDim d);


  void cudaD_binarize_probs(dim3 Gr, dim3 Bl, double* states, const double* probs, double* rand, MatrixDim d);

}


#endif
