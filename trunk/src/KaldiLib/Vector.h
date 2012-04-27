//
// C++ Interface: %{MODULE}
//
// Description:
//
//
// Author: %{AUTHOR} <%{EMAIL}>, (C) %{YEAR}
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef TNet_Vector_h
#define TNet_Vector_h

#include <cstddef>
#include <cstdlib>
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

namespace TNet
{
  template<typename _ElemT> class Vector;
  template<typename _ElemT> class SubVector;
  template<typename _ElemT> class Matrix;
  template<typename _ElemT> class SpMatrix;

  // we need to declare the friend functions here
  template<typename _ElemT>
    std::ostream & operator << (std::ostream & rOut, const Vector<_ElemT> & rV);

  template<typename _ElemT>
    std::istream & operator >> (std::istream & rIn, Vector<_ElemT> & rV);

  template<typename _ElemT>
    _ElemT
    BlasDot(const Vector<_ElemT>& rA, const Vector<_ElemT>& rB);

  /** **************************************************************************
   ** **************************************************************************
   *  @brief Provides a matrix abstraction class
   *
   *  This class provides a way to work with matrices in TNet.
   *  It encapsulates basic operations and memory optimizations.
   *
   */
  template<typename _ElemT>
    class Vector
    {
    public:

    /// defines a type of this
    typedef Vector<_ElemT> ThisType;


    Vector(): mpData(NULL)
#ifdef STK_MEMALIGN_MANUAL
    ,mpFreeData(NULL)
#endif
    , mDim(0)
      {}

      /**
       * @brief Copy constructor
       * @param rV
       */
      Vector(const Vector<_ElemT>& rV)
  	  { mpData=NULL; Init(rV.Dim()); Copy(rV); }


      /* Type conversion constructor. */
      template<typename _ElemU>
      explicit Vector(const Vector<_ElemU>& rV)
  	  { mpData=NULL; Init(rV.Dim()); Copy(rV); }


      Vector(const _ElemT* ppData, const size_t s)
      { mpData=NULL; Init(s); Copy(ppData); }

      explicit Vector(const size_t s, bool clear=true)
      { mpData=NULL; Init(s,clear); }

      ~Vector()
      { Destroy(); }

       Vector<_ElemT> &operator = (const Vector <_ElemT> &other)
       { Init(other.Dim()); Copy(other); return *this; } // Needed for inclusion in std::vector

      Vector<_ElemT>&
      Init(size_t length, bool clear=true);

      /**
       * @brief Dealocates the window from memory and resets the dimensions to (0)
       */
      void
      Destroy();

      /**
       * @brief Returns @c true if vector is initialized
       */
      bool
      IsInitialized() const
      { return mpData != NULL; }

      /**
       * @brief Sets all elements to 0
       */
      void
      Zero();

      void
      Set(_ElemT f);

      inline size_t
      Dim() const
      { return mDim; }

      /**
       * @brief Returns size of matrix in memory (in bytes)
       */
      inline size_t
      MSize() const
      {
        return (mDim + (((16 / sizeof(_ElemT)) - mDim%(16 / sizeof(_ElemT)))
                          % (16 / sizeof(_ElemT)))) * sizeof(_ElemT);
      }

      /**
       *  @brief Gives access to the vector memory area
       *  @return pointer to the first field
       */
      inline _ElemT*
      pData()
      { return mpData; }

      /**
       *  @brief Gives access to the vector memory area
       *  @return pointer to the first field (const version)
       */
      inline const _ElemT*
      pData() const
      { return mpData; }

      /**
       *  @brief Gives access to a specified vector element (const).
       */
      inline _ElemT
      operator [] (size_t i) const
      {
#ifdef PARANOID
		assert(i<mDim);
#endif
		return *(mpData + i);
	  }

      /**
       *  @brief Gives access to a specified vector element (non-const).
       */
      inline _ElemT &
      operator [] (size_t i)
      {
#ifdef PARANOID
		assert(i<mDim);
#endif
		return *(mpData + i);
	  }

      /**
       *  @brief Gives access to a specified vector element (const).
       */
      inline _ElemT
		operator () (size_t i) const
      {
#ifdef PARANOID
		assert(i<mDim);
#endif
		return *(mpData + i);
	  }

      /**
       *  @brief Gives access to a specified vector element (non-const).
       */
      inline _ElemT &
		operator () (size_t i)
      {
#ifdef PARANOID
		assert(i<mDim);
#endif
		return *(mpData + i);
	  }

      /**
       * @brief Returns a matrix sub-range
       * @param o Origin
       * @param l Length
       * See @c SubVector class for details
       */
      SubVector<_ElemT>
      Range(const size_t o, const size_t l)
      { return SubVector<_ElemT>(*this, o, l); }

      /**
       * @brief Returns a matrix sub-range
       * @param o Origin
       * @param l Length
       * See @c SubVector class for details
       */
      const SubVector<_ElemT>
      Range(const size_t o, const size_t l) const
      { return SubVector<_ElemT>(*this, o, l); }



      //########################################################################
      //########################################################################

      /// Copy data from another vector
      Vector<_ElemT>&
      Copy(const Vector<_ElemT>& rV);

      /// Copy data from another vector of a different type.
      template<typename _ElemU> Vector<_ElemT>&
      Copy(const Vector<_ElemU>& rV);


      /// Load data into the vector
      Vector<_ElemT>&
      Copy(const _ElemT* ppData);

      Vector<_ElemT>&
      CopyVectorizedMatrixRows(const Matrix<_ElemT> &rM);

      Vector<_ElemT>&
      RemoveElement(size_t i);

      Vector<_ElemT>&
      ApplyLog();

      Vector<_ElemT>&
      ApplyLog(const Vector<_ElemT>& rV);//ApplyLog to rV and put the result in (*this)

      Vector<_ElemT>&
      ApplyExp();

      Vector<_ElemT>&
      ApplySoftMax();

      Vector<_ElemT>&
      Invert();

      Vector<_ElemT>&
      DotMul(const Vector<_ElemT>& rV); // Multiplies each element (*this)(i) by rV(i).

      Vector<_ElemT>&
      BlasAxpy(const _ElemT alpha, const Vector<_ElemT>& rV);

      Vector<_ElemT>&
      BlasGemv(const _ElemT alpha, const Matrix<_ElemT>& rM, const MatrixTrasposeType trans, const Vector<_ElemT>& rV, const _ElemT beta = 0.0);


      //########################################################################
      //########################################################################

      Vector<_ElemT>&
      Add(const Vector<_ElemT>& rV)
      { return BlasAxpy(1.0, rV); }

      Vector<_ElemT>&
      Subtract(const Vector<_ElemT>& rV)
      { return BlasAxpy(-1.0, rV); }

      Vector<_ElemT>&
      AddScaled(_ElemT alpha, const Vector<_ElemT>& rV)
      { return BlasAxpy(alpha, rV); }

      Vector<_ElemT>&
      Add(_ElemT c);

      Vector<_ElemT>&
      MultiplyElements(const Vector<_ElemT>& rV);

      // @brief elementwise : rV.*rR+beta*this --> this
      Vector<_ElemT>&
      MultiplyElements(_ElemT alpha, const Vector<_ElemT>& rV, const Vector<_ElemT>& rR,_ElemT beta);

      Vector<_ElemT>&
      DivideElements(const Vector<_ElemT>& rV);

      /// @brief elementwise : rV./rR+beta*this --> this
      Vector<_ElemT>&
      DivideElements(_ElemT alpha, const Vector<_ElemT>& rV, const Vector<_ElemT>& rR,_ElemT beta);

      Vector<_ElemT>&
      Subtract(_ElemT c);

      Vector<_ElemT>&
      Scale(_ElemT c);


      //########################################################################
      //########################################################################

      /// Performs a row stack of the matrix rMa
      Vector<_ElemT>&
      MatrixRowStack(const Matrix<_ElemT>& rMa);

      // Extracts a row of the matrix rMa.  .. could also do this with vector.Copy(rMa[row]).
      Vector<_ElemT>&
      Row(const Matrix<_ElemT>& rMa, size_t row);

      // Extracts a column of the matrix rMa.
      Vector<_ElemT>&
      Col(const Matrix<_ElemT>& rMa, size_t col);

      // Takes all elements to a power.
      Vector<_ElemT>&
      Power(_ElemT power);

      _ElemT 
      Max() const;

      _ElemT 
      Min() const;

      /// Returns sum of the elements
      _ElemT
      Sum() const;

      /// Returns sum of the elements
      Vector<_ElemT>&
      AddRowSum(const Matrix<_ElemT>& rM);

      /// Returns sum of the elements
      Vector<_ElemT>&
      AddColSum(const Matrix<_ElemT>& rM);

      /// Returns log(sum(exp())) without exp overflow
      _ElemT
      LogSumExp() const;

      //########################################################################
      //########################################################################

      friend std::ostream &
      operator << <> (std::ostream& rOut, const Vector<_ElemT>& rV);

      friend _ElemT
      BlasDot<>(const Vector<_ElemT>& rA, const Vector<_ElemT>& rB);

      /**
       * Computes v1^T * M * v2.  
       * Not as efficient as it could be where v1==v2 (but no suitable blas
       * routines available).
       */
      _ElemT
      InnerProduct(const Vector<_ElemT> &v1, const Matrix<_ElemT> &M, const Vector<_ElemT> &v2) const;


    //##########################################################################
    //##########################################################################
    //protected:
    public:
      /// data memory area
      _ElemT*   mpData;
#ifdef STK_MEMALIGN_MANUAL
      /// data to be freed (in case of manual memalignment use, see common.h)
      _ElemT*   mpFreeData;
#endif
      size_t  mDim;      ///< Number of elements
    }; // class Vector




  /**
   * @brief Represents a non-allocating general vector which can be defined
   * as a sub-vector of higher-level vector
   */
  template<typename _ElemT>
    class SubVector : public Vector<_ElemT>
    {
    protected:
      /// Constructor
      SubVector(const Vector<_ElemT>& rT,
                const size_t  origin,
                const size_t  length)
      {
        assert(origin+length <= rT.mDim);
        Vector<_ElemT>::mpData = rT.mpData+origin;
        Vector<_ElemT>::mDim   = length;
      }
      //only Vector class can call this protected constructor
      friend class Vector<_ElemT>; 

    public:
      /// Constructor
      SubVector(Vector<_ElemT>& rT,
                const size_t  origin,
                const size_t  length)
      {
        assert(origin+length <= rT.mDim);
        Vector<_ElemT>::mpData = rT.mpData+origin;
        Vector<_ElemT>::mDim   = length;
      }


      /**
       * @brief Constructs a vector representation out of a standard array
       *
       * @param pData pointer to data array to associate with this vector
       * @param length length of this vector
       */
      inline
      SubVector(_ElemT *ppData,
                size_t length)
      {
        Vector<_ElemT>::mpData = ppData;
        Vector<_ElemT>::mDim   = length;
      }


      /**
       * @brief Destructor
       */
      ~SubVector()
      {
        Vector<_ElemT>::mpData = NULL;
      }
    };


    // Useful shortcuts
    typedef Vector<BaseFloat> BfVector;
    typedef SubVector<BaseFloat> BfSubVector;

    //Adding two vectors of different types
    template <typename _ElemT, typename _ElemU>
    void Add(Vector<_ElemT>& rDst, const Vector<_ElemU>& rSrc)
    {
      assert(rDst.Dim() == rSrc.Dim());
      const _ElemU* p_src = rSrc.pData();
      _ElemT* p_dst = rDst.pData();

      for(size_t i=0; i<rSrc.Dim(); i++) {
        *p_dst++ += (_ElemT)*p_src++;
      }
    }
   
      
    //Scales adding two vectors of different types
    template <typename _ElemT, typename _ElemU>
    void AddScaled(Vector<_ElemT>& rDst, const Vector<_ElemU>& rSrc, _ElemT scale)
    {
      assert(rDst.Dim() == rSrc.Dim());

      Vector<_ElemT> tmp(rSrc);
      rDst.BlasAxpy(scale, tmp); 

/*
      const _ElemU* p_src = rSrc.pData();
      _ElemT* p_dst = rDst.pData();

      for(size_t i=0; i<rDst.Dim(); i++) {
        *p_dst++ += *p_src++ * scale;
      }
*/
    }


} // namespace TNet

//*****************************************************************************
//*****************************************************************************
// we need to include the implementation
#include "Vector.tcc"

/******************************************************************************
 ******************************************************************************
 * The following section contains specialized template definitions
 * whose implementation is in Vector.cc
 */


#endif // #ifndef TNet_Vector_h
