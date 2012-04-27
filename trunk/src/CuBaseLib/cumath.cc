


#include "cumath.h"
#include "cukernels.h"


namespace TNet {

  //////////////////////////////////////////////////////////////////////////////
  //// CuMath<> Template specializations (float)
  ////
  template<>
  void CuMath<float>::Sigmoid(CuMatrix<float>& Y, const CuMatrix<float>& X)
  {
    Timer tim; tim.Start();

    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(X.Cols(),CUBLOCK), n_blocks(X.Rows(), CUBLOCK));

    cudaF_sigmoid(dimGrid, dimBlock, Y.pCUData(), X.pCUData(), X.Dim());
    cuSafeCall(cudaGetLastError());
    
    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }

  template<>
  void CuMath<float>::DiffSigmoid(CuMatrix<float>& Eout, const CuMatrix<float>& Ein, const CuMatrix<float>& Y)
  {
    Timer tim; tim.Start();

    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Eout.Cols(), CUBLOCK), n_blocks(Eout.Rows(),CUBLOCK));

    cudaF_diff_sigmoid(dimGrid, dimBlock, Eout.pCUData(), Ein.pCUData(), Y.pCUData(), Eout.Dim());
    cuSafeCall(cudaGetLastError());

    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }

    
  template<>
  void CuMath<float>::Softmax(CuMatrix<float>& Y, const CuMatrix<float>& X)
  {
    Timer tim; tim.Start();

#if 0
    //disable 'reduce' functions
    size_t dimBlock = CUBLOCK;
    size_t dimGrid  = n_blocks(X.Rows(),CUBLOCK);

    cudaF_softmax(dimGrid, dimBlock, Y.pCUData(), X.pCUData(), X.Dim());
    cuSafeCall(cudaGetLastError());
#else
    if(X.Cols() > 256) {
      //use old implementation (can't use reduction due to 
      //limited size of shared memory)
      size_t dimBlock = CUBLOCK;
      size_t dimGrid  = n_blocks(X.Rows(),CUBLOCK);

      cudaF_softmax(dimGrid, dimBlock, Y.pCUData(), X.pCUData(), X.Dim());
      cuSafeCall(cudaGetLastError());
    } else {
      //use implementation with reduction
      dim3 dimBlock(X.Cols(),1);
      dim3 dimGrid(1,X.Rows());

      cudaF_softmax_reduce(dimGrid, dimBlock, Y.pCUData(), X.pCUData(), X.Dim());
      cuSafeCall(cudaGetLastError());
    }
#endif

    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }



  template<>
  void CuMath<float>::BlockLinearity(CuMatrix<float>& Y, const CuMatrix<float>& X, const CuMatrix<float>& block_transf)
  {
    Timer tim; tim.Start();

    assert(Y.Rows() == X.Rows());
    assert((X.Cols() % block_transf.Rows()) == 0);
    assert((Y.Cols() % block_transf.Cols()) == 0);
    assert((X.Cols() / block_transf.Rows()) == (Y.Cols() / block_transf.Cols()));

    int blocks = X.Cols() / block_transf.Rows();

    for(int i = 0; i < blocks; i++) {
      int m = block_transf.Cols();
      int n = X.Rows();
      int k = block_transf.Rows();

      /*
      //DEBUG MESSAGE
      std::cout << "N N " << m << " " << n << " " << k << " " 
                << 1.0 << " " << block_transf << " " << block_transf.Stride() 
                << " " << X+i*k << " " << X.Stride() << " " 
                << 0.0 << " " << Y+i*n << " " << Y.Stride() 
                << "\n" << std::flush;
      */


      cublasSgemm('N', 'N', m, n, k, 
                  1.0, block_transf.pCUData(), block_transf.Stride(), 
                  X.pCUData()+i*k, X.Stride(), 
                  0.0, Y.pCUData()+i*m, Y.Stride());
    }
    cuSafeCall(cublasGetError());    
    
    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }




  template<>
  void CuMath<float>::Expand(CuMatrix<float>& Y, const CuMatrix<float>& X, const CuVector<int>& frameOffsets)
  {
    Timer tim; tim.Start();

    assert(Y.Rows() == X.Rows());
    assert(X.Cols() * frameOffsets.Dim() == Y.Cols());

    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Y.Cols(), CUBLOCK), n_blocks(Y.Rows(),CUBLOCK));

    cudaF_expand(dimGrid, dimBlock, Y.pCUData(), X.pCUData(), frameOffsets.pCUData(), Y.Dim(), X.Dim());
    cuSafeCall(cudaGetLastError());
    
    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }


  template<>
  void CuMath<float>::Rearrange(CuMatrix<float>& Y, const CuMatrix<float>& X, const CuVector<int>& copyFrom)
  {
    Timer tim; tim.Start();

    assert(copyFrom.Dim() == Y.Cols());
    assert(Y.Rows() == X.Rows());
    
    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Y.Cols(), CUBLOCK), n_blocks(Y.Rows(),CUBLOCK));
    
    cudaF_rearrange(dimGrid, dimBlock, Y.pCUData(), X.pCUData(), copyFrom.pCUData(), Y.Dim(), X.Dim());
    cuSafeCall(cudaGetLastError());
    
    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }



  template<>
  void CuMath<float>::Randomize(CuMatrix<float>& Y, const CuMatrix<float>& X, const CuVector<int>& copyFrom)
  {
    Timer tim; tim.Start();

    assert(X.Cols() == Y.Cols());
    assert(X.Rows() == Y.Rows());
    assert(copyFrom.Dim() <= Y.Rows());
    
    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Y.Cols(), CUBLOCK), n_blocks(copyFrom.Dim(),CUBLOCK));
    
    MatrixDim dimX = X.Dim(); dimX.rows=copyFrom.Dim();
    MatrixDim dimY = Y.Dim(); dimY.rows=copyFrom.Dim();

    cudaF_randomize(dimGrid, dimBlock, Y.pCUData(), X.pCUData(), copyFrom.pCUData(), dimY, dimX);
    cuSafeCall(cudaGetLastError());

    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }



  template<>
  void CuMath<float>::CheckClass(const CuMatrix<float>& out, const CuMatrix<float> &des, CuVector<int>& match)
  {
    Timer tim; tim.Start();

    assert(out.Cols() == des.Cols());
    assert(out.Rows() == des.Rows());
    assert(out.Stride() == des.Stride());
    assert(match.Dim() == out.Rows());

    if(out.Cols() > 256) {
      size_t dimBlock = CUBLOCK;
      size_t dimGrid = n_blocks(out.Rows(),CUBLOCK);

      cudaF_check_class(dimGrid, dimBlock, out.pCUData(), des.pCUData(), match.pCUData(), out.Dim());
      cuSafeCall(cudaGetLastError());
    } else {
      dim3 dimBlock(out.Cols(),1);
      dim3 dimGrid(1,out.Rows());

      cudaF_check_class_reduce(dimGrid, dimBlock, out.pCUData(), des.pCUData(), match.pCUData(), out.Dim());
      cuSafeCall(cudaGetLastError());
    }



    
    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }


  template<>
  void CuMath<float>::OffsetGemm(char transA, char transB, float alpha, const CuMatrix<float>& A, const CuMatrix<float>& B, float beta, CuMatrix<float>& C, int offA, int offB, int offC)
  {
    Timer tim; tim.Start();
    // CUBLAS is col major, TNet is row major
    // keep trans..., just swap A&B argumets: A->B B->A
    //
    // WARNING
    // NO DIMENSION CHECK!!!
    
    //m,n,k is cublas m,n,k
    size_t m = ((transB=='T' || transB=='t')? B.Rows() : B.Cols()); 
    size_t n = ((transA=='T' || transA=='t')? A.Cols() : A.Rows());
    size_t k = ((transB=='T' || transB=='t')? B.Cols() : B.Rows());
    size_t k1 = ((transA=='T' || transA=='t')? A.Rows() : A.Cols());

    k = ((k<k1)?k:k1);
    m = ((m<C.Cols())?m:C.Cols());
    n = ((n<C.Rows())?m:C.Rows());

#if 0
    std::cout << "A " << transA << " "<< A.Rows() << " " << A.Cols() << " " << A.Stride() << " " << offA
         << "; B " << transB << " "<< B.Rows() << " " << B.Cols() << " " << B.Stride() << " " << offB
         << "; C " << C.Rows() << " " << C.Cols() << " " << C.Stride() << " " << offC
         << "; alpha" << alpha << " beta" << beta << " REALmnk:" << m <<" "<< n <<" "<< k << std::endl;
#endif
         

    cublasSgemm(transB, transA, m, n, k, 
                alpha, B.pCUData()+offB, B.Stride(), 
                A.pCUData()+offA, A.Stride(), 
                beta, C.pCUData()+offC, C.Stride());
    cuSafeCall(cublasGetError());    
    
    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }


/*

  template<>
  void CuMath<float>::Gemv(char trans, float alpha, const CuMatrix<float>& A, const float* x, size_t dimX, float beta, float* y, size_t dimY)
  {
    Timer tim; tim.Start();
    // CUBLAS is col major, TNet is row major
    // y = alpha * op(A) * x + beta * y,
    
    size_t m = A.Cols(); //m..rows of A in colmajor (== cols in rowmajor)
    size_t n = A.Rows(); //n..cols of A in colmajor (== rows in rowmajor)
 
    // switch the trans parameter!
    char cu_trans;
    if(trans == 't' || trans == 'T') {
      cu_trans = 'n';
    } else if (trans == 'n' || trans == 'N') {
      cu_trans = 't';
    } else {
      Error(std::string("Unknown trans")+trans);
    }
   
    //check the dims
    if(cu_trans == 'n') {
      assert(dimX == n);
      assert(dimY == m);
    } else {
      assert(dimX == m);
      assert(dimY == n);
    }
 
    //run gemv
    cublasSgemv(cu_trans,m,n,alpha,
                A.pCUData(), A.Stride(), x, 1,
                beta, y, 1);
    
    cuSafeCall(cublasGetError());    
    
    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }
  */

  /**
   * offsetY tells how many outputs of 'Ax' mutiplication is skipped at the beginning, 
   */
  template<>
  void CuMath<float>::OffsetGemv(char trans, float alpha, const CuMatrix<float>& A, const float* x, size_t dimX, float beta, float* y, size_t dimY, size_t offsetY)
  {
    Timer tim; tim.Start();
    // CUBLAS is col major, TNet is row major
    // y = alpha * op(A) * x + beta * y,
    
    size_t m = A.Cols(); //m..rows of A in colmajor (== cols in rowmajor)
    size_t n = A.Rows(); //n..cols of A in colmajor (== rows in rowmajor)
 
    // switch the trans parameter!
    char cu_trans;
    if(trans == 't' || trans == 'T') {
      cu_trans = 'n';
    } else if (trans == 'n' || trans == 'N') {
      cu_trans = 't';
    } else {
      Error(std::string("Unknown trans")+trans);
    }

    // select part of matrix for compute
    size_t cu_offset = 0;
    if(cu_trans == 'n') {
      cu_offset += offsetY;
      assert(m >= dimY+offsetY);
      m = dimY;
    } else {
      cu_offset += offsetY*A.Stride();
      assert(n >= dimY+offsetY);
      n = dimY;
    }
   
    //check the dims
    if(cu_trans == 'n') {
      assert(dimX == n);
      assert(dimY == m);
    } else {
      assert(dimX == m);
      assert(dimY == n);
    }
 
    //run gemv
    cublasSgemv(cu_trans,m,n,alpha,
                A.pCUData()+cu_offset, A.Stride(), x, 1,
                beta, y, 1);
    
    cuSafeCall(cublasGetError());    
    
    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }

  
  template<>
  void CuMath<float>::BlasGer(float alpha, const float* x, size_t dimX, const float* y, size_t dimY, CuMatrix<float>& A) {
    Timer tim; tim.Start();
    // CUBLAS is col major, TNet is row major
    // -> switch x and y

    // A = alpha * x * transpose(y) + A,
    
    assert(dimX == A.Rows());
    assert(dimY == A.Cols());

    size_t m = A.Cols(); //m..rows of A in colmajor (== cols in rowmajor)
    size_t n = A.Rows(); //n..cols of A in colmajor (== rows in rowmajor)

    cublasSger(m,n,alpha,y,1,x,1,A.pCUData(),A.Stride()); 
    cuSafeCall(cublasGetError());    

    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }


  template<>
  void CuMath<float>::VecExpand(const CuVector<float>&in, CuVector<float>&out)
  {
    Timer tim; tim.Start();

    assert(out.Dim() % in.Dim() == 0);
    int n_copies = out.Dim()/in.Dim();
    CuVector<int> offsets(n_copies);
    //offsets.SetConst(0); done implicitly!

    dim3 dimBlock(CUBLOCK);
    dim3 dimGrid(n_blocks(out.Dim(), CUBLOCK));
    
    MatrixDim dim_in = { 1, in.Dim(), in.Dim() };
    MatrixDim dim_out = { 1, out.Dim(), out.Dim() };
    cudaF_expand(dimGrid, dimBlock, out.pCUData(), in.pCUData(), offsets.pCUData(), dim_out, dim_in);
    cuSafeCall(cudaGetLastError());

    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }


  template<>
  void CuMath<float>::VecAddColSum(float alpha, const CuVector<float>&in, float beta, CuVector<float>&out)
  {
    Timer tim; tim.Start();

    assert(in.Dim() % out.Dim() == 0);

    size_t dimBlock = CUBLOCK;
    size_t dimGrid = n_blocks(out.Dim(),CUBLOCK); 

    MatrixDim dim = { in.Dim()/out.Dim(), out.Dim(), out.Dim() };

    cudaF_add_col_sum(dimGrid,dimBlock,alpha,in.pCUData(),beta,out.pCUData(),dim);

    cuSafeCall(cudaGetLastError());
    
    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }


  //////////////////////////////////////////////////////////////////////////////
  //// CuMath<> Template specializations (double)
  ////
  template<>
  void CuMath<double>::Sigmoid(CuMatrix<double>& Y, const CuMatrix<double>& X)
  {
    Timer tim; tim.Start();

    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(X.Cols(),CUBLOCK), n_blocks(X.Rows(), CUBLOCK));

    cudaD_sigmoid(dimGrid, dimBlock, Y.pCUData(), X.pCUData(), X.Dim());
    cuSafeCall(cudaGetLastError());
    
    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }

  template<>
  void CuMath<double>::DiffSigmoid(CuMatrix<double>& Eout, const CuMatrix<double>& Ein, const CuMatrix<double>& Y)
  {
    Timer tim; tim.Start();

    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Eout.Cols(), CUBLOCK), n_blocks(Eout.Rows(),CUBLOCK));

    cudaD_diff_sigmoid(dimGrid, dimBlock, Eout.pCUData(), Ein.pCUData(), Y.pCUData(), Eout.Dim());
    cuSafeCall(cudaGetLastError());

    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }

    
  template<>
  void CuMath<double>::Softmax(CuMatrix<double>& Y, const CuMatrix<double>& X)
  {
    Timer tim; tim.Start();

    size_t dimBlock = CUBLOCK;
    size_t dimGrid  = n_blocks(X.Rows(),CUBLOCK);

    cudaD_softmax(dimGrid, dimBlock, Y.pCUData(), X.pCUData(), X.Dim());
    cuSafeCall(cudaGetLastError());

    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }



  template<>
  void CuMath<double>::BlockLinearity(CuMatrix<double>& Y, const CuMatrix<double>& X, const CuMatrix<double>& block_transf)
  {
    Timer tim; tim.Start();

    assert(Y.Rows() == X.Rows());
    assert((X.Cols() % block_transf.Rows()) == 0);
    assert((Y.Cols() % block_transf.Cols()) == 0);
    assert((X.Cols() / block_transf.Rows()) == (Y.Cols() / block_transf.Cols()));

    int blocks = X.Cols() / block_transf.Rows();

    for(int i = 0; i < blocks; i++) {
      int m = block_transf.Cols();
      int n = X.Rows();
      int k = block_transf.Rows();

      /*
      //DEBUG MESSAGE
      std::cout << "N N " << m << " " << n << " " << k << " " 
                << 1.0 << " " << block_transf << " " << block_transf.Stride() 
                << " " << X+i*k << " " << X.Stride() << " " 
                << 0.0 << " " << Y+i*n << " " << Y.Stride() 
                << "\n" << std::flush;
      */


      cublasDgemm('N', 'N', m, n, k, 
                  1.0, block_transf.pCUData(), block_transf.Stride(), 
                  X.pCUData()+i*k, X.Stride(), 
                  0.0, Y.pCUData()+i*m, Y.Stride());
    }
    cuSafeCall(cublasGetError());    
    
    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }




  template<>
  void CuMath<double>::Expand(CuMatrix<double>& Y, const CuMatrix<double>& X, const CuVector<int>& frameOffsets)
  {
    Timer tim; tim.Start();

    assert(Y.Rows() == X.Rows());
    assert(X.Cols() * frameOffsets.Dim() == Y.Cols());

    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Y.Cols(), CUBLOCK), n_blocks(Y.Rows(),CUBLOCK));

    cudaD_expand(dimGrid, dimBlock, Y.pCUData(), X.pCUData(), frameOffsets.pCUData(), Y.Dim(), X.Dim());
    cuSafeCall(cudaGetLastError());
    
    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }


  template<>
  void CuMath<double>::Rearrange(CuMatrix<double>& Y, const CuMatrix<double>& X, const CuVector<int>& copyFrom)
  {
    Timer tim; tim.Start();

    assert(copyFrom.Dim() == Y.Cols());
    assert(Y.Rows() == X.Rows());
    
    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Y.Cols(), CUBLOCK), n_blocks(Y.Rows(),CUBLOCK));
    
    cudaD_rearrange(dimGrid, dimBlock, Y.pCUData(), X.pCUData(), copyFrom.pCUData(), Y.Dim(), X.Dim());
    cuSafeCall(cudaGetLastError());
    
    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }



  template<>
  void CuMath<double>::Randomize(CuMatrix<double>& Y, const CuMatrix<double>& X, const CuVector<int>& copyFrom)
  {
    Timer tim; tim.Start();

    assert(X.Cols() == Y.Cols());
    assert(X.Rows() == Y.Rows());
    assert(copyFrom.Dim() <= Y.Rows());
    
    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Y.Cols(), CUBLOCK), n_blocks(copyFrom.Dim(),CUBLOCK));
    
    MatrixDim dimX = X.Dim(); dimX.rows=copyFrom.Dim();
    MatrixDim dimY = Y.Dim(); dimY.rows=copyFrom.Dim();

    cudaD_randomize(dimGrid, dimBlock, Y.pCUData(), X.pCUData(), copyFrom.pCUData(), dimY, dimX);
    cuSafeCall(cudaGetLastError());

    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }



  template<>
  void CuMath<double>::CheckClass(const CuMatrix<double>& out, const CuMatrix<double> &des, CuVector<int>& match)
  {
    Timer tim; tim.Start();

    assert(out.Cols() == des.Cols());
    assert(out.Rows() == des.Rows());
    assert(out.Stride() == des.Stride());
    assert(match.Dim() == out.Rows());

    size_t dimBlock = CUBLOCK;
    size_t dimGrid = n_blocks(out.Rows(),CUBLOCK);

    cudaD_check_class(dimGrid, dimBlock, out.pCUData(), des.pCUData(), match.pCUData(), out.Dim());
    cuSafeCall(cudaGetLastError());
    
    tim.End(); CuDevice::Instantiate().AccuProfile(__func__,tim.Val());
  }

}
