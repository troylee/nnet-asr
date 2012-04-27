
#include "cuActivation.h"
#include "cumath.h"


namespace TNet {


  void
  CuSigmoid::
  PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    CuMath<BaseFloat>::Sigmoid(Y, X);
  }


  void 
  CuSigmoid::
  BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    CuMath<BaseFloat>::DiffSigmoid(Y, X, mOutput);
  }



  void 
  CuSoftmax::
  PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    CuMath<BaseFloat>::Softmax(Y,X);
  }

   
   
  void
  CuSoftmax::
  BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    //we assume X is already dE/dSoftmax_input
    Y.CopyFrom(X);
  }



} //namespace

