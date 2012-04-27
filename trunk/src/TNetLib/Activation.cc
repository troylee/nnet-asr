
#include "Activation.h"


namespace TNet {

void Sigmoid::PropagateFnc(const BfMatrix& X, BfMatrix& Y) {
  //Y = 1/(1+e^{-X})
  for(size_t r=0; r<X.Rows(); r++) {
    for(size_t c=0; c<X.Cols(); c++) {
      Y(r,c) = 1.0f/(1.0f+exp(-X(r,c)));
    }
  }
}


void Sigmoid::BackpropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y) {
  const Matrix<BaseFloat>& out = GetOutput();
  //Y = OUT*(1-OUT)*X //ODVOZENO
  for(size_t r=0; r<X.Rows(); r++) {
    for(size_t c=0; c<X.Cols(); c++) {
      Y(r,c) = X(r,c)*out(r,c)*(1.0f-out(r,c));
    }
  }
}



void Softmax::PropagateFnc(const BfMatrix& X, BfMatrix& Y) {
  //Y_j = e^X_j / sum_i(e^X_i)
  //
  //    e^(X_j+c) / sum_i(e^X_i+c)
  //    = e^c.e^X_h / e^c.sum_i(e^X_i)
  //    = e^X_j / sum_i(e^X_i)
  //
  size_t rows = X.Rows();
  for(size_t i=0; i<rows; i++) {
    BfSubVector y_i(Y[i]); //<< y_i gets pointer to i'th row of matrix Y
    y_i.Copy(X[i]);
    BaseFloat max = y_i.Max();
    y_i.Subtract(max);
    y_i.ApplyExp();
    BaseFloat sum = y_i.Sum();
    y_i.Scale(1.0f/sum);
  }
}


void Softmax::BackpropagateFnc(const BfMatrix& X, BfMatrix& Y) {
  //simply copy the error...,
  Y.Copy(X);
}


void BlockSoftmax::ReadFromStream(std::istream& rIn) {
  rIn >> mDim; 
  mDimOffset.Init(mDim.Dim()+1);
  
  int off=0; 
  for(int i=0; i<mDim.Dim(); i++) { 
    mDimOffset[i]=off;
    off += mDim[i];
  }
  mDimOffset[mDim.Dim()]=off;

  if(off!=GetNOutputs()) {
    KALDI_ERR << "Non-matching dimension of sum of softmaxes,"
      << " the sum:" << off 
      << " GetNOutputs:" << GetNOutputs();
  }
}

void BlockSoftmax::WriteToStream(std::ostream& rOut) {
  rOut << mDim;
}




void BlockSoftmax::PropagateFnc(const BfMatrix& X, BfMatrix& Y) {
  //Y_j = e^X_j / sum_i(e^X_i)
  //
  //    e^(X_j+c) / sum_i(e^X_i+c)
  //    = e^c.e^X_h / e^c.sum_i(e^X_i)
  //    = e^X_j / sum_i(e^X_i)
  //
  size_t rows = X.Rows();
  for(size_t i=0; i<rows; i++) {
    BfSubVector y_i(Y[i]); //<< y_i gets pointer to i'th row of matrix Y
    y_i.Copy(X[i]);
    //BaseFloat max = y_i.Max();
    //y_i.Subtract(max);
    //y_i.ApplyExp();
    //normalize separately on each softmax interval...
    for(int j=0; j<mDim.Dim(); j++) {
      BfSubVector y_i_smx_j(y_i.Range(mDimOffset[j],mDim[j]));
      BaseFloat max = y_i_smx_j.Max();
      y_i_smx_j.Subtract(max);
      y_i_smx_j.ApplyExp();
      BaseFloat sum = y_i_smx_j.Sum();
      y_i_smx_j.Scale(1.0f/sum);
    }
  }

//  X.CheckData("BlockSoftmax PropagateFnc X");
//  Y.CheckData("BlockSoftmax PropagateFnc Y");
}


void BlockSoftmax::BackpropagateFnc(const BfMatrix& X, BfMatrix& Y) {
  //set the output to zero
  Y.Zero();
  //copy only parts of the error
  //from softmax intervals which sum up to 0.0, not 1.0
  for(int i=0; i<X.Rows(); i++) {
    for(int j=0; j<mDim.Dim(); j++) {
      const BfSubVector x_i_smx_j(X[i].Range(mDimOffset[j],mDim[j]));
      BaseFloat sum = x_i_smx_j.Sum();
      if(sum > -0.1 && sum < 0.1) {
        BfSubVector y_i_smx_j(Y[i].Range(mDimOffset[j],mDim[j]));
        y_i_smx_j.Copy(x_i_smx_j);
      } else if (sum > 0.9 && sum < 1.1) {
        ; //do nothing
      } else {
        KALDI_ERR << "Invalid sum: " << sum;
      }
    }
  }

//  X.CheckData("BlockSoftmax BackpropagateFnc X");
//  Y.CheckData("BlockSoftmax BackpropagateFnc Y");

}



} //namespace TNet

