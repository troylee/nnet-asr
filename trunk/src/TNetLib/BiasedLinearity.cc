

#include "BiasedLinearity.h"


namespace TNet {


void
BiasedLinearity::
PropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y)
{
  //y = b + x.A

  //precopy bias
  size_t rows = X.Rows();
  for(size_t i=0; i<rows; i++) {
    Y[i].Copy(*mpBias);
  }

  //multiply matrix by matrix with mLinearity
  Y.BlasGemm(1.0f, X, NO_TRANS, *mpLinearity, NO_TRANS, 1.0f);
}


void
BiasedLinearity::
BackpropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y)
{
  // e' = e.A^T
  Y.Zero();
  Y.BlasGemm(1.0f, X, NO_TRANS, *mpLinearity, TRANS, 0.0f);
}



void
BiasedLinearity::
ReadFromStream(std::istream& rIn)
{
  //matrix is stored transposed as SNet does
  Matrix<BaseFloat> transpose;
  rIn >> transpose;
  mLinearity = Matrix<BaseFloat>(transpose, TRANS);
  //biases stored normally
  rIn >> mBias;
}

 
void
BiasedLinearity::
WriteToStream(std::ostream& rOut)
{
  //matrix is stored transposed as SNet does
  Matrix<BaseFloat> transpose(mLinearity, TRANS);
  rOut << transpose;
  //biases stored normally
  rOut << mBias;
  rOut << std::endl;
}


void
BiasedLinearity::
Gradient()
{
  //calculate gradient of weight matrix
  mLinearityCorrection.Zero();
  mLinearityCorrection.BlasGemm(1.0f, GetInput(), TRANS, 
                                GetErrorInput(), NO_TRANS, 
                                0.0f);

  //calculate gradient of bias
  mBiasCorrection.Set(0.0f);
  size_t rows = GetInput().Rows();
  for(size_t i=0; i<rows; i++) {
    mBiasCorrection.Add(GetErrorInput()[i]);
  }

  /* 
  //perform update
  mLinearity.AddScaled(-mLearningRate, mLinearityCorrection);
  mBias.AddScaled(-mLearningRate, mBiasCorrection);
  */
}


void 
BiasedLinearity::
AccuGradient(const UpdatableComponent& src, int thr, int thrN) {
  //cast the argument
  const BiasedLinearity& src_comp = dynamic_cast<const BiasedLinearity&>(src);

  //allocate accumulators when needed
  if(mLinearityCorrectionAccu.MSize() == 0) {
    mLinearityCorrectionAccu.Init(mLinearity.Rows(),mLinearity.Cols());
  }
  if(mBiasCorrectionAccu.MSize() == 0) {
    mBiasCorrectionAccu.Init(mBias.Dim());
  }

  //need to find out which rows to sum...
  int div = mLinearityCorrection.Rows() / thrN;
  int mod = mLinearityCorrection.Rows() % thrN;

  int origin = thr * div + ((mod > thr)? thr : mod);
  int rows = div + ((mod > thr)? 1 : 0);

  //create the matrix windows
  const SubMatrix<BaseFloat> src_mat (
    src_comp.mLinearityCorrection, 
    origin, rows, 
    0, mLinearityCorrection.Cols()
  );
  SubMatrix<double> tgt_mat (
    mLinearityCorrectionAccu, 
    origin, rows, 
    0, mLinearityCorrection.Cols()
  );
  //sum the rows
  Add(tgt_mat,src_mat);

  //first thread will always sum the bias correction
  if(thr == 0) {
    Add(mBiasCorrectionAccu,src_comp.mBiasCorrection);
  }

}


void
BiasedLinearity::
Update(int thr, int thrN)
{
  //need to find out which rows to sum...
  int div = mLinearity.Rows() / thrN;
  int mod = mLinearity.Rows() % thrN;

  int origin = thr * div + ((mod > thr)? thr : mod);
  int rows = div + ((mod > thr)? 1 : 0);

  //std::cout << "[P" << thr << "," << origin << "," << rows << "]" << std::flush;

  //get the matrix windows
  SubMatrix<double> src_mat (
    mLinearityCorrectionAccu, 
    origin, rows, 
    0, mLinearityCorrection.Cols()
  );
  SubMatrix<BaseFloat> tgt_mat (
    mLinearity, 
    origin, rows, 
    0, mLinearityCorrection.Cols()
  );


  //update weights
  AddScaled(tgt_mat, src_mat, -mLearningRate);

  //perform L2 regularization (weight decay)
  BaseFloat L2_decay = -mLearningRate * mWeightcost * mBunchsize;
  if(L2_decay != 0.0) {
    tgt_mat.AddScaled(L2_decay, tgt_mat);
  }

  //first thread always update bias
  if(thr == 0) {
    //std::cout << "[" << thr << "BP]" << std::flush;
    AddScaled(mBias, mBiasCorrectionAccu, -mLearningRate);
  }

  //reset the accumulators
  src_mat.Zero();
  if(thr == 0) {
    mBiasCorrectionAccu.Zero();
  }

}

} //namespace
