#ifndef _CUVECTOR_H_
#define _CUVECTOR_H_

#include "Vector.h"

namespace TNet {

  template<typename _ElemT> class CuMatrix;

  /**
   * Matrix for CUDA computing
   */
  template<typename _ElemT>
  class CuVector 
  {
    typedef CuVector<_ElemT> ThisType;

    public:

      /// Default Constructor
      CuVector<_ElemT>()
       : mDim(0), mpCUData(NULL)
      { }
      /// Constructor with memory initialisation
      CuVector<_ElemT>(size_t dim)
       : mDim(0), mpCUData(NULL)
      { Init(dim); }

      /// Destructor
      ~CuVector()
      { Destroy(); }

      /// Dimensions
      size_t Dim() const
      { return mDim; }

      /*
      ::MatrixDim Dim() const
      { ::MatrixDim d = { mDim, 1, 1 }; return d; }
      */

      /// Get raw pointer
      const _ElemT* pCUData() const
      { return mpCUData; }
      _ElemT* pCUData()
      { return mpCUData; }

      /// Allocate the memory
      ThisType& Init(size_t dim);

      /// Deallocate the memory
      void Destroy();

      /// Copy functions (reallocates when needed)
      ThisType&        CopyFrom(const CuVector<_ElemT>& rSrc);
      ThisType&        CopyFrom(const Vector<_ElemT>& rSrc);
      Vector<_ElemT>&  CopyTo(Vector<_ElemT>& rDst) const;


      
      // Math operations
      //
      void SetZero();

      void SetConst(_ElemT value)
      { Error("__func__ Not implemented"); }

      void AddScaled(_ElemT alpha, const CuVector<_ElemT>& vec, _ElemT beta)
      { Error("__func__ Not implemented"); }

      void AddColSum(_ElemT alpha, const CuMatrix<_ElemT>& mat, _ElemT beta)
      { Error("__func__ Not implemented"); }

      void Print() const
      { 
        Vector<_ElemT> vec(Dim());
        CopyTo(vec);
        std::cout << vec << "\n";
      }


    private:
      size_t mDim;
      _ElemT* mpCUData;
  };


  /// Prints the matrix dimensions and pointer to stream
  template<typename _ElemT>
  inline std::ostream& operator << (std::ostream& out, const CuVector<_ElemT>& vec)
  { 
    size_t d = vec.Dim(); 
    out << "[CuVector D" << d
        << " PTR" << vec.pCUData() << "]" << std::flush;
    return out;
  }
  
  
}


#include "cuvector.tcc"

#endif
