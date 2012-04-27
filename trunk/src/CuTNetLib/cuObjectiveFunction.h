#ifndef _CUOBJ_FUN_I_
#define _CUOBJ_FUN_I_

#include <cassert>
#include <limits>
#include <cmath>
#include <sstream>

#include "Vector.h"
#include "cuvector.h"
#include "cumatrix.h"

namespace TNet 
{

  
  /**
   * General interface for objective functions
   */
  class CuObjectiveFunction
  {
    public:
      /// Enum with objective function types
      typedef enum { 
        OBJ_FUN_I = 0x0300, 
        MEAN_SQUARE_ERROR, 
        CROSS_ENTROPY, 
      } ObjFunType;

      /// Factory for creating objective function instances
      static CuObjectiveFunction* Factory(ObjFunType type);
    
    //////////////////////////////////////////////////////////////
    // Interface specification
    public:
      CuObjectiveFunction() 
      { } 

      virtual ~CuObjectiveFunction() 
      { }

      virtual ObjFunType GetTypeId() = 0; 
      virtual const char* GetTypeLabel() = 0; 

      /// evaluates the data, calculate global error
      virtual void Evaluate(const CuMatrix<BaseFloat>& rNetOutput, const CuMatrix<BaseFloat>& rDesired, CuMatrix<BaseFloat>& rNetError) = 0;
 
      ///get the average per frame error
      virtual double GetError() = 0;  
      ///the number of processed frames 
      virtual size_t GetFrames() = 0;
      ///report the error to std::cout 
      virtual std::string Report() = 0;
  };




  /**
   * Means square error, useful for autoencoders, RBMs et al.
   */
  class CuMeanSquareError : public CuObjectiveFunction
  {
    public:
      CuMeanSquareError() 
        : mError(0), mFrames(0)
      { }
    
      virtual ~CuMeanSquareError() 
      { }

      ObjFunType GetTypeId()
      { return CuObjectiveFunction::MEAN_SQUARE_ERROR; }

      const char* GetTypeLabel()
      { return "<mean_square_error>"; }

      void Evaluate(const CuMatrix<BaseFloat>& rNetOutput, const CuMatrix<BaseFloat>& rDesired, CuMatrix<BaseFloat>& rNetError);
      
      double GetError()
      { return mError; }  
      
      size_t GetFrames()
      { return mFrames; }
      
      std::string Report()
      { 
        std::ostringstream ss;
        ss << "Mse:" << mError << " frames:" << mFrames 
           << " err/frm:" << mError/mFrames << "\n";
        return ss.str();
      }

    private:
      double mError;
      size_t mFrames;

      CuMatrix<BaseFloat> mAuxMat;
      CuVector<BaseFloat> mAuxVec;
      Vector<BaseFloat> mAuxVecHost;

  };


 /**
   * Cross entropy, it assumes desired vectors as output values 
   */
  class CuCrossEntropy : public CuObjectiveFunction
  {
    public:
      CuCrossEntropy() 
        : mError(0), mFrames(0), mCorrect(0)
      { }
      
      ~CuCrossEntropy() 
      { }
      
      ObjFunType GetTypeId()
      { return CuObjectiveFunction::CROSS_ENTROPY; }

      const char* GetTypeLabel()
      { return "<cross_entropy>"; }

      void Evaluate(const CuMatrix<BaseFloat>& rNetOutput, const CuMatrix<BaseFloat>& rDesired, CuMatrix<BaseFloat>& rNetError);

      double GetError()
      { return mError; }

      size_t GetFrames()
      { return mFrames; }

      std::string Report()
      {
        std::ostringstream ss;
        //for compatibility with SNet
        //ss << " correct: >> " << 100.0*mCorrect/mFrames << "% <<\n";
        
        //current new format...
        ss << "Xent:" << mError << " frames:" << mFrames 
           << " err/frm:" << mError/mFrames 
           << " correct[" << 100.0*mCorrect/mFrames << "%]"
           << "\n";
        return ss.str();
      }

    private:
      double mError;
      size_t mFrames;
      size_t mCorrect;
      
      CuMatrix<BaseFloat> mAuxMat;
      CuVector<BaseFloat> mAuxVec;
      Vector<BaseFloat> mAuxVecHost;

      CuVector<int> mClassifyVec;
      Vector<int> mClassifyVecHost;
  };





} //namespace TNet


#endif
