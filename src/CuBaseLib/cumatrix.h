#ifndef _CUMATRIX_H_
#define _CUMATRIX_H_

#include <sstream>

#include "Matrix.h"
#include "cukernels.h"



namespace TNet {

  template<typename _ElemT> class CuVector;

  /**
   * Matrix for CUDA computing
   */
  template<typename _ElemT>
  class CuMatrix 
  {
    typedef CuMatrix<_ElemT> ThisType;

    public:

      /// Default Constructor
      CuMatrix<_ElemT>()
       : mRows(0), mCols(0), mStride(0), mpCUData(NULL)
      { }
      /// Constructor with memory initialisation
      CuMatrix<_ElemT>(size_t rows, size_t cols)
       : mRows(0), mCols(0), mStride(0), mpCUData(NULL)
      { Init(rows, cols); }

      /// Destructor
      ~CuMatrix()
      { Destroy(); }

      /// Dimensions
      size_t Rows() const
      { return mRows; }

      size_t Cols() const 
      { return mCols; }

      size_t Stride() const
      { return mStride; }

      ::MatrixDim Dim() const
      { ::MatrixDim d = { 
          static_cast<int>(mRows), 
          static_cast<int>(mCols), 
          static_cast<int>(mStride) 
        }; 
        return d; 
      }

      /// Get raw pointer
      const _ElemT* pCUData() const
      { return mpCUData; }
      _ElemT* pCUData()
      { return mpCUData; }

      /// Get raw row pointer
      const _ElemT* pCURowData(size_t r) const
      { assert(r < Rows()); return mpCUData+r*mStride; }
      _ElemT* pCURowData(size_t r)
      { assert(r < Rows()); return mpCUData+r*mStride; }

      /// Get size of matrix in bytes
      size_t MSize() const
      { return mRows*mStride*sizeof(_ElemT); }
      /// Get size of matrix row in bytes
      size_t MRowSize() const
      { return mStride*sizeof(_ElemT); }

      /// Allocate the memory
      ThisType& Init(size_t rows, size_t cols);

      /// Deallocate the memory
      void Destroy();

      /// Copy functions (reallocates when needed)
      ThisType&        CopyFrom(const CuMatrix<_ElemT>& rSrc);
      ThisType&        CopyFrom(const Matrix<_ElemT>& rSrc);
      Matrix<_ElemT>&  CopyTo(Matrix<_ElemT>& rDst) const;

      /// Copy rowCnt rows from rSrc, starting by row srcOri, 
      /// copying to memory block starting by row dstOri
      void CopyRows(size_t rowCnt, size_t srcOri, const CuMatrix<_ElemT>& rSrc, size_t dstOri);

      /// Copy colCnt columns from rSrc, starting by col srcOri, 
      /// copying to memory block starting by row dstOri
      void CopyCols(size_t colCnt, size_t srcOri, const CuMatrix<_ElemT>& rSrc, size_t dstOri);


      // Math operations, some calling kernels
      //
      void SetZero();

      void SetConst(_ElemT value)
      { Error("__func__ Not implemented"); }

      void ApplyLog()
      { Error("__func__ Not implemented"); }

      void ApplyMask(const CuMatrix<BaseFloat>& mask)
      { Error("__func__ Not implemented"); }

      void ApplyL1(BaseFloat l1)
      { Error("__func__ Not implemented"); }

      /// scale i'th column by scale[i]
      void ScaleCols(const CuVector<_ElemT>& scale)
      { Error("__func__ Not implemented"); }

      /// scale i'th row by scale[i]
      void ScaleRows(const CuVector<_ElemT>& scale)
      { Error("__func__ Not implemented"); }

      /// B = aplha * A + beta * B
      void AddScaled(_ElemT alpha, const CuMatrix<_ElemT>& A, _ElemT beta)
      { Error("__func__ Not implemented"); }

      /// B = aplha * row + beta * B
      void AddScaledRow(_ElemT alpha, const CuVector<_ElemT>& row, _ElemT beta)
      { Error("__func__ Not implemented"); }

      /// C = alpha * A(^T)*B(^T) + beta * C
      void Gemm(char transa, char transb, 
                _ElemT alpha, 
                const CuMatrix<_ElemT>& A, const CuMatrix<_ElemT>& B, 
                _ElemT beta)
      { Error("__func__ Not implemented"); }

      /// A = alpha * x*y^T + A
      void BlasGer(_ElemT alpha, 
                const CuVector<_ElemT>& x, const CuVector<_ElemT>& y)
      { Error("__func__ Not implemented"); }


      /// Multiply two matrices elementhwise: C = A .* C
      void MulElem(const CuMatrix<_ElemT>& A)
      { Error("__func__ Not implemented"); }
      
      /// A = log(A)
      void LogElem()
      { Error("__func__ Not implemented"); }

      void Print() const
      { 
        Matrix<_ElemT> mat(Rows(),Cols());
        CopyTo(mat);
        std::cout << mat;
      }

      

      void CheckData()
      {
        Matrix<_ElemT> mat;
        CopyTo(mat);
        for(size_t i=0; i<Rows(); i++) {
          for(size_t j=0; j<Cols(); j++) {
            if(std::isnan(mat(i,j)) || std::isinf(mat(i,j))) {
              std::ostringstream os;
              os << "Invalid value:" << mat(i,j) << "at row"<<i<<" col"<<j<<"\n";
              Error(os.str());
            }
          }
        }
      }
        
      
    private:
      size_t mRows;
      size_t mCols;
      size_t mStride;

      _ElemT* mpCUData;

  };


  /// Prints the matrix dimensions and pointer to stream
  template<typename _ElemT>
  inline std::ostream& operator << (std::ostream& out, const CuMatrix<_ElemT>& mat)
  { 
    out << "[CUMATRIX R" << mat.Rows() << " C" << mat.Cols() << " S" << mat.Stride() 
        << " PTR" << mat.pCUData() << "]" << std::flush;
    return out;
  }
  
  
}


#include "cumatrix.tcc"

#endif
