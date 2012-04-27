#ifndef TNet_Matrix_h
#define TNet_Matrix_h

#include <stddef.h>
#include <stdlib.h>
#include <stdexcept>
#include <iostream>

#ifdef HAVE_ATLAS
extern "C"{
  #include <cblas.h>
  #include <clapack.h>
}
#endif

#include "Common.h"
#include "MathAux.h"
#include "Types.h"
#include "Error.h"

//#define TRACE_MATRIX_OPERATIONS
#define CHECKSIZE

namespace TNet
{


  //  class matrix_error : public std::logic_error {};
  //  class matrix_sizes_error : public matrix_error {};

  // declare the class so the header knows about it
  template<typename _ElemT> class Vector;
  template<typename _ElemT> class SubVector;
  template<typename _ElemT> class Matrix;
  template<typename _ElemT> class SubMatrix;

  // we need to declare the friend << operator here
  template<typename _ElemT>
    std::ostream & operator << (std::ostream & rOut, const Matrix<_ElemT> & rM);

  // we need to declare the friend << operator here
  template<typename _ElemT>
    std::istream & operator >> (std::istream & rIn, Matrix<_ElemT> & rM);

  // we need to declare this friend function here
  template<typename _ElemT>
   _ElemT TraceOfProduct(const Matrix<_ElemT> &A, const Matrix<_ElemT> &B); // tr(A B)

  // we need to declare this friend function here
  template<typename _ElemT> 
	 _ElemT TraceOfProductT(const Matrix<_ElemT> &A, const Matrix<_ElemT> &B); // tr(A B^T)==tr(A^T B)


  /** **************************************************************************
   ** **************************************************************************
   *  @brief Provides a matrix class
   *
   *  This class provides a way to work with matrices in TNet.
   *  It encapsulates basic operations and memory optimizations.
   *
   */
  template<typename _ElemT>
    class Matrix
    {
    public:
      /// defines a transpose type

      struct HtkHeader
      {
        INT_32    mNSamples;              
        INT_32    mSamplePeriod;
        INT_16    mSampleSize;
        UINT_16   mSampleKind;
      };


      /** 
       * @brief Extension of the HTK header
       */
      struct HtkHeaderExt
      {
        INT_32 mHeaderSize;
        INT_32 mVersion;
        INT_32 mSampSize;
      };  




      /// defines a type of this
      typedef Matrix<_ElemT>    ThisType;

      // Constructors

      /// Empty constructor
      Matrix<_ElemT> ():
        mpData(NULL), mMCols(0), mMRows(0), mStride(0)
#ifdef STK_MEMALIGN_MANUAL
        , mpFreeData(NULL)
#endif
      {}

      /// Copy constructor
      Matrix<_ElemT> (const Matrix<_ElemT> & rM, MatrixTrasposeType trans=NO_TRANS):
        mpData(NULL) 
      { if(trans==NO_TRANS){ Init(rM.mMRows, rM.mMCols); Copy(rM); } else { Init(rM.mMCols,rM.mMRows); Copy(rM,TRANS); } }

      /// Copy constructor from another type.
      template<typename _ElemU>
      explicit Matrix<_ElemT> (const Matrix<_ElemU> & rM, MatrixTrasposeType trans=NO_TRANS):
        mpData(NULL) 
      { if(trans==NO_TRANS){ Init(rM.Rows(), rM.Cols()); Copy(rM); } else { Init(rM.Cols(),rM.Rows()); Copy(rM,TRANS); } }

      /// Basic constructor
      Matrix(const size_t r, const size_t c, bool clear=true)
      { mpData=NULL; Init(r, c, clear);  }


	  Matrix<_ElemT> &operator = (const Matrix <_ElemT> &other) { Init(other.Rows(), other.Cols()); Copy(other); return *this; } // Needed for inclusion in std::vector

      /// Destructor
      ~Matrix()
      { Destroy(); }


      /// Initializes matrix (if not done by constructor)
      ThisType &
      Init(const size_t r,
           const size_t c, bool clear=true);

      /**
       * @brief Dealocates the matrix from memory and resets the dimensions to (0, 0)
       */
      void
      Destroy();


      ThisType &
      Zero();

      ThisType &
      Unit(); // set to unit.

      /** 
       * @brief Copies the contents of a matrix
       * @param rM Source data matrix
       * @return Returns reference to this
       */
      template<typename _ElemU> ThisType &
        Copy(const Matrix<_ElemU> & rM, MatrixTrasposeType Trans=NO_TRANS);



      /**
       * @brief Copies the elements of a vector row-by-row into a matrix
       * @param rV Source vector
       * @param nRows Number of rows of returned matrix
       * @param nCols Number of columns of returned matrix
       *
       * Note that rV.Dim() must equal nRows*nCols
       */
      ThisType &
      CopyVectorSplicedRows(const Vector<_ElemT> &rV, const size_t nRows, const size_t nCols);

      /**
       * @brief Returns @c true if matrix is initialized
       */
      bool
		IsInitialized() const
      { return mpData != NULL; }

      /// Returns number of rows in the matrix
      inline size_t
		Rows() const
      {
        return mMRows;
      }

      /// Returns number of columns in the matrix
      inline size_t
      Cols() const
      {
        return mMCols;
      }

      /// Returns number of columns in the matrix memory
      inline size_t
		Stride() const
      {
        return mStride;
      }


      /**
       *  @brief Gives access to a specified matrix row without range check
       *  @return Pointer to the const array
       */
      inline const _ElemT*  __attribute__((aligned(16)))
       pData () const
      {
        return mpData;
      }


      /**
       *  @brief Gives access to a specified matrix row without range check
       *  @return Pointer to the non-const data array
       */
      inline _ElemT* __attribute__((aligned(16)))
       pData () 
      {
        return mpData;
      }


      /**
       *  @brief pData_workaround is a workaround that allows SubMatrix to get a 
       *  @return pointer to non-const data even though the Matrix is const... 
       */
    protected:
      inline _ElemT*  __attribute__((aligned(16)))
       pData_workaround () const
      {
        return mpData;
      }
    public:


      /// Returns size of matrix in memory
      size_t
      MSize() const
      {
        return mMRows * mStride * sizeof(_ElemT);
      }

      /// Checks the content of the matrix for nan and inf values
      void
      CheckData(const std::string file = "") const
      {
        for(size_t row=0; row<Rows(); row++) {
          for(size_t col=0; col<Cols(); col++) {
            if(isnan((*this)(row,col)) || isinf((*this)(row,col))) {
              std::ostringstream os;
              os << "Invalid value: " << (*this)(row,col)
                 << " in matrix row: " << row
                 << " col: " << col
                 << " file: " << file;
              Error(os.str());
            }
          }
        }
      }

      /**
       *  **********************************************************************
       *  **********************************************************************
       *  @defgroup RESHAPE Matrix reshaping rutines
       *  **********************************************************************
       *  **********************************************************************
       * @{
       */

      /**
       *  @brief Removes one row from the matrix. The memory is not reallocated.
       */
      ThisType &
      RemoveRow(size_t i);      

      /** @} */

      /**
       *  **********************************************************************
       *  **********************************************************************
       *  @defgroup ACCESS Access functions and operators
       *  **********************************************************************
       *  **********************************************************************
       * @{
       */

      /**
       *  @brief Gives access to a specified matrix row without range check
       *  @return Subvector object representing the row
       */
      inline const SubVector<_ElemT>
      operator []  (size_t i) const
      {
        assert(i < mMRows);
        return SubVector<_ElemT>(mpData + (i * mStride), Cols());
      }

      inline SubVector<_ElemT>
      operator []  (size_t i)
      {
        assert(i < mMRows);
        return SubVector<_ElemT>(mpData + (i * mStride), Cols());
      }

      /**
       *  @brief Gives access to a specified matrix row without range check
       *  @return pointer to the first field of the row
       */
      inline  _ElemT*
      pRowData(size_t i) 
      {
        assert(i < mMRows);
        return mpData + i * mStride;
      }

      /**
       *  @brief Gives access to a specified matrix row without range check
       *  @return pointer to the first field of the row (const version)
       */
      inline const _ElemT*
      pRowData(size_t i) const
      {
        assert(i < mMRows);
        return mpData + i * mStride;
      }

      /**
       *  @brief Gives access to matrix elements (row, col)
       *  @return reference to the desired field
       */
      inline _ElemT&
		operator () (size_t r, size_t c)
      { 
#ifdef PARANOID
        assert(r < mMRows && c < mMCols);
#endif
		return *(mpData + r * mStride + c); 
	  }

      /**
       *  @brief Gives access to matrix elements (row, col)
       *  @return pointer to the desired field (const version)
       */
      inline const _ElemT
		operator () (size_t r, size_t c) const
      { 
#ifdef PARANOID
        assert(r < mMRows && c < mMCols);
#endif
		return *(mpData + r * mStride + c); 
	  }

      /**
       * @brief Returns a matrix sub-range
       * @param ro Row offset
       * @param r  Rows in range
       * @param co Column offset
       * @param c  Coluns in range
       * See @c SubMatrix class for details
       */
      SubMatrix<_ElemT>
      Range(const size_t    ro, const size_t    r,
            const size_t    co, const size_t    c)
      { return SubMatrix<_ElemT>(*this, ro, r, co, c); }

      const SubMatrix<_ElemT>
      Range(const size_t    ro, const size_t    r,
            const size_t    co, const size_t    c) const
      { return SubMatrix<_ElemT>(*this, ro, r, co, c); }
      /** @} */


      /**
       *  **********************************************************************
       *  **********************************************************************
       *  @defgroup MATH ROUTINES
       *  **********************************************************************
       *  **********************************************************************
       *  @{
       **/

      /**
       *  @brief Returns sum of all elements
       */
      _ElemT&
      Sum() const;

      ThisType &
      DotMul(const ThisType& a);

      ThisType &
      Scale(_ElemT alpha);

      ThisType &
      ScaleCols(const Vector<_ElemT> &scale); // Equivalent to (*this) = (*this) * diag(scale).

      ThisType &
      ScaleRows(const Vector<_ElemT> &scale); // Equivalent to (*this) = diag(scale) * (*this);

      /// Sum another matrix rMatrix with this matrix
      ThisType&
      Add(const Matrix<_ElemT>& rMatrix);

   
      /// Sum scaled matrix rMatrix with this matrix
      ThisType&
      AddScaled(_ElemT alpha, const Matrix<_ElemT>& rMatrix);

      /// Apply log to all items of the matrix
      ThisType&
      ApplyLog();

      /**
       * @brief Computes the determinant of this matrix
       * @return Returns the determinant of a matrix
       * @ingroup MATH
       *
       */
      _ElemT LogAbsDeterminant(_ElemT *DetSign=NULL);


      /**
       *  @brief Performs matrix inplace inversion
       */
      ThisType &
      Invert(_ElemT *LogDet=NULL, _ElemT *DetSign=NULL, bool inverse_needed=true);

      /**
       *  @brief Performs matrix inplace inversion in double precision, even if this object is not double precision.
       */
      ThisType &
      InvertDouble(_ElemT *LogDet=NULL, _ElemT *DetSign=NULL, bool inverse_needed=true){
        double LogDet_tmp, DetSign_tmp;
        Matrix<double> dmat(*this); dmat.Invert(&LogDet_tmp, &DetSign_tmp, inverse_needed); if(inverse_needed) (*this).Copy(dmat); 
        if(LogDet) *LogDet = LogDet_tmp; if(DetSign) *DetSign = DetSign_tmp;
        return *this;
      }


      /**
       *  @brief Inplace matrix transposition. Applicable only to square matrices
       */
      ThisType &
      Transpose()
      {
        assert(Rows()==Cols());
        size_t M=Rows();
        for(size_t i=0;i<M;i++)
          for(size_t j=0;j<i;j++){
           _ElemT &a = (*this)(i,j), &b = (*this)(j,i);
		   std::swap(a,b);
        }
		return *this;
      }


      


      bool IsSymmetric(_ElemT cutoff = 1.0e-05) const;

      bool IsDiagonal(_ElemT cutoff = 1.0e-05) const;

      bool IsUnit(_ElemT cutoff = 1.0e-05) const;

      bool IsZero(_ElemT cutoff = 1.0e-05) const;

      _ElemT FrobeniusNorm() const; // sqrt of sum of square elements.

      _ElemT LargestAbsElem() const; // largest absolute value.

	
      friend _ElemT TNet::TraceOfProduct<_ElemT>(const Matrix<_ElemT> &A, const Matrix<_ElemT> &B); // tr(A B)
      friend _ElemT TNet::TraceOfProductT<_ElemT>(const Matrix<_ElemT> &A, const Matrix<_ElemT> &B); // tr(A B^T)==tr(A^T B)
      friend class SubMatrix<_ElemT>; // so it can get around const restrictions on the pointer to mpData.

      /** **********************************************************************
       *  **********************************************************************
       *  @defgroup BLAS_ROUTINES BLAS ROUTINES
       *  @ingroup MATH
       *  **********************************************************************
       *  **********************************************************************
       **/

      ThisType &
      BlasGer(const _ElemT alpha, const Vector<_ElemT>& rA, const Vector<_ElemT>& rB);

      ThisType &
	    Axpy(const _ElemT alpha, const Matrix<_ElemT> &rM, MatrixTrasposeType transA=NO_TRANS);

      ThisType &
      BlasGemm(const _ElemT alpha,
               const ThisType& rA, MatrixTrasposeType transA,
               const ThisType& rB, MatrixTrasposeType transB,
               const _ElemT beta = 0.0);


      /** @} */


      /** **********************************************************************
       *  **********************************************************************
       *  @defgroup IO Input/Output ROUTINES
       *  **********************************************************************
       *  **********************************************************************
       *  @{
       **/

      friend std::ostream &
      operator << <> (std::ostream & out, const ThisType & m);
	  	
      void PrintOut(char *file);
      void ReadIn(char *file);


      bool
      LoadHTK(const char* pFileName);

      /** @} */


    protected:
//      inline void swap4b(void *a);
//      inline void swap2b(void *a);


    protected:
      /// data memory area
      _ElemT*   mpData;

      /// these atributes store the real matrix size as it is stored in memory
      /// including memalignment
      size_t    mMCols;       ///< Number of columns
      size_t    mMRows;       ///< Number of rows
      size_t    mStride;      ///< true number of columns for the internal matrix.
                              ///< This number may differ from M_cols as memory
                              ///< alignment might be used

#ifdef STK_MEMALIGN_MANUAL
      /// data to be freed (in case of manual memalignment use, see Common.h)
      _ElemT*   mpFreeData;
#endif
    }; // class Matrix

    template<>  Matrix<float> &  Matrix<float>::Invert(float *LogDet, float *DetSign, bool inverse_needed); // state that we will implement separately for float and double.
    template<>  Matrix<double> &  Matrix<double>::Invert(double *LogDet, double *DetSign, bool inverse_needed);



  /** **************************************************************************
   ** **************************************************************************
   *  @brief Sub-matrix representation
   *
   *  This class provides a way to work with matrix cutouts in STK.
   *
   *
   */
  template<typename _ElemT>
    class SubMatrix : public Matrix<_ElemT>
    {
    typedef SubMatrix<_ElemT>    ThisType;

    public:
      /// Constructor
      SubMatrix(const Matrix<_ElemT>& rT, // Input matrix cannot be const because SubMatrix can change its contents.
                const size_t    ro,
                const size_t    r,
                const size_t    co,
                const size_t    c);


      /// The destructor
      ~SubMatrix<_ElemT>()
      {
#ifndef STK_MEMALIGN_MANUAL
        Matrix<_ElemT>::mpData = NULL;
#else
        Matrix<_ElemT>::mpFreeData = NULL;
#endif
      }

      /// Assign operator
      ThisType& operator=(const ThisType& rSrc)
      {
        //std::cout << "[PERFORMing operator= SubMatrix&^2]" << std::flush;
        this->mpData = rSrc.mpData;
        this->mMCols = rSrc.mMCols;
        this->mMRows = rSrc.mMRows;
        this->mStride = rSrc.mStride;
        this->mpFreeData = rSrc.mpFreeData;
        return *this;
      }

   

      /// Initializes matrix (if not done by constructor)
      ThisType &
      Init(const size_t r,
           const size_t c, bool clear=true)
      { Error("Submatrix cannot do Init"); return *this; }

      /**
       * @brief Dealocates the matrix from memory and resets the dimensions to (0, 0)
       */
      void
      Destroy()
      { Error("Submatrix cannot do Destroy"); }



    };



  //Create useful shortcuts
  typedef Matrix<BaseFloat> BfMatrix;
  typedef SubMatrix<BaseFloat> BfSubMatrix;

  /**
   * Function for summing matrices of different types
   */
  template<typename _ElemT, typename _ElemU>
  void Add(Matrix<_ElemT>& rDst,  const Matrix<_ElemU>& rSrc) {
    assert(rDst.Cols() == rSrc.Cols());
    assert(rDst.Rows() == rSrc.Rows());

    for(size_t i=0; i<rDst.Rows(); i++) {
      const _ElemU* p_src = rSrc.pRowData(i);
      _ElemT* p_dst = rDst.pRowData(i);
      for(size_t j=0; j<rDst.Cols(); j++) {
        *p_dst++ += (_ElemT)*p_src++;
      }
    }
  }

  /**
   * Function for summing matrices of different types
   */
  template<typename _ElemT, typename _ElemU>
  void AddScaled(Matrix<_ElemT>& rDst, const Matrix<_ElemU>& rSrc, _ElemT scale) {
    assert(rDst.Cols() == rSrc.Cols());
    assert(rDst.Rows() == rSrc.Rows());

    Vector<_ElemT> tmp(rDst[0]);

    for(size_t i=0; i<rDst.Rows(); i++) {
      tmp.Copy(rSrc[i]);
      rDst[i].BlasAxpy(scale, tmp);

      /*
      const _ElemU* p_src = rSrc.pRowData(i);
      _ElemT* p_dst = rDst.pRowData(i);
      for(size_t j=0; j<rDst.Cols(); j++) {
        *p_dst++ += (_ElemT)(*p_src++) * scale;
      }
      */
    }
  }





} // namespace STK



//*****************************************************************************
//*****************************************************************************
// we need to include the implementation
#include "Matrix.tcc"
//*****************************************************************************
//*****************************************************************************


/******************************************************************************
 ******************************************************************************
 * The following section contains specialized template definitions
 * whose implementation is in Matrix.cc
 */


//#ifndef TNet_Matrix_h
#endif
