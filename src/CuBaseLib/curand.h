#ifndef _CU_RAND_H_
#define _CU_RAND_H_


#include "cumatrix.h"


namespace TNet {
  
  template<typename T> 
  class CuRand {
   public:

    CuRand(size_t rows, size_t cols)
    { SeedGpu(rows,cols); }

    ~CuRand() { }

    void SeedGpu(size_t rows, size_t cols);
    void Rand(CuMatrix<T>& tgt);
    void GaussRand(CuMatrix<T>& tgt);

    void BinarizeProbs(const CuMatrix<T>& probs, CuMatrix<T>& states);
    void AddGaussNoise(CuMatrix<T>& tgt, T gscale = 1.0);
  
   private:
    static void SeedRandom(Matrix<unsigned>& mat);
     
   private:
    CuMatrix<unsigned> z1, z2, z3, z4;
    CuMatrix<T> tmp;
  };

}


#include "curand.tcc"


#endif
