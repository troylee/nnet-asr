
#include "cuObjectiveFunction.h"

#include "Error.h"
#include "cumath.h"


namespace TNet
{



  CuObjectiveFunction*
  CuObjectiveFunction::
  Factory(ObjFunType type) {
    CuObjectiveFunction* ret = NULL;
    switch(type) {
      case MEAN_SQUARE_ERROR:  ret = new CuMeanSquareError;  break;
      case CROSS_ENTROPY:      ret = new CuCrossEntropy;     break;
      default: Error("Unknown ObjFun type");
    }
    return ret;
  }


  void 
  CuMeanSquareError::
  Evaluate(const CuMatrix<BaseFloat>& rNetOutput, const CuMatrix<BaseFloat>& rDesired, CuMatrix<BaseFloat>& rNetError)
  {
    //get the global error
    rNetError.CopyFrom(rNetOutput);
    rNetError.AddScaled(-1.0,rDesired,1.0);

    //calculate the MSE
    mAuxMat.CopyFrom(rNetError);
    mAuxMat.MulElem(mAuxMat);
    
    mAuxVec.Init(mAuxMat.Cols());
    mAuxVec.AddColSum(1.0,mAuxMat,0.0);
    mAuxVec.CopyTo(mAuxVecHost);

    mError += mAuxVecHost.Sum();
 
    //count the frames    
    mFrames += rNetError.Rows();
  }

  void 
  CuCrossEntropy::
  Evaluate(const CuMatrix<BaseFloat>& rNetOutput, const CuMatrix<BaseFloat>& rDesired, CuMatrix<BaseFloat>& rNetError)
  {
    if(rDesired.Cols() != rNetOutput.Cols()) {
      std::ostringstream os;
      os << "Non-matching dimensions of network output with training targets!!!"
         << " Netoutput:" << rNetOutput.Cols()
         << " Targets:" << rDesired.Cols();
      Error(os.str());
    }

    //get the global error
    //dXent/dSoftmax_in = y-d
    rNetError.CopyFrom(rNetOutput);
    rNetError.AddScaled(-1.0,rDesired,1.0);
   
    //check classification
    mClassifyVec.Init(rNetOutput.Rows());
    CuMath<BaseFloat>::CheckClass(rNetOutput,rDesired,mClassifyVec);
    mClassifyVec.CopyTo(mClassifyVecHost);
    mCorrect += mClassifyVecHost.Sum();

    //calculate Xent
    mAuxMat.CopyFrom(rNetOutput);
    mAuxMat.LogElem();
    mAuxMat.MulElem(rDesired);

    mAuxVec.Init(mAuxMat.Cols());
    mAuxVec.AddColSum(-1.0,mAuxMat,0.0);
    mAuxVec.CopyTo(mAuxVecHost);

    mError += mAuxVecHost.Sum();

    //count the frames    
    mFrames += rNetError.Rows();
  }


} // namespace TNet
