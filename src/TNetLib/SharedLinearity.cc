

#include "SharedLinearity.h"
#include "cblas.h"

namespace TNet {

void 
SharedLinearity::
PropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y)
{
  //precopy bias
  for(int k=0; k<mNInstances; k++) {
    for(size_t r=0; r<X.Rows(); r++) {
      memcpy(Y.pRowData(r)+k*mpBias->Dim(),mpBias->pData(),mpBias->Dim()*sizeof(BaseFloat));
    }
  }
  
  //multiply blockwise
  for(int k=0; k<mNInstances; k++) {
    SubMatrix<BaseFloat> xblock(X,0,X.Rows(),k*mpLinearity->Rows(),mpLinearity->Rows());
    SubMatrix<BaseFloat> yblock(Y,0,Y.Rows(),k*mpLinearity->Cols(),mpLinearity->Cols());
    yblock.BlasGemm(1.0,xblock,NO_TRANS,*mpLinearity,NO_TRANS,1.0);
  }
}


void 
SharedLinearity::
BackpropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y)
{
  for(int k=0; k<mNInstances; k++) {
    SubMatrix<BaseFloat> xblock(X,0,X.Rows(),k*mpLinearity->Cols(),mpLinearity->Cols());
    SubMatrix<BaseFloat> yblock(Y,0,Y.Rows(),k*mpLinearity->Rows(),mpLinearity->Rows());
    yblock.BlasGemm(1.0,xblock,NO_TRANS,*mpLinearity,TRANS,1.0);
  }
}

#if 0
void 
SharedLinearity::
AccuUpdate() 
{
  BaseFloat N = 1;
  /* 
  //Not part of the interface!!!
  if(mGradDivFrm) {
    N = static_cast<BaseFloat>(GetInput().Rows());
  }
  */
  BaseFloat mmt_gain = static_cast<BaseFloat>(1.0/(1.0-mMomentum));
  N *= mmt_gain; //compensate higher gradient estimates due to momentum 
  
  //compensate augmented dyn. range of gradient caused by multiple instances
  N *= static_cast<BaseFloat>(mNInstances); 

  const Matrix<BaseFloat>& X = GetInput().Data();
  const Matrix<BaseFloat>& E = GetErrorInput().Data();
  //get gradient of shared linearity
  for(int k=0; k<mNInstances; k++) {
    SubMatrix<BaseFloat> xblock(X,0,X.Rows(),k*mLinearity.Rows(),mLinearity.Rows());
    SubMatrix<BaseFloat> eblock(E,0,E.Rows(),k*mLinearity.Cols(),mLinearity.Cols());
    mLinearityCorrection.BlasGemm(1.0,xblock,TRANS,eblock,NO_TRANS,((k==0)?mMomentum:1.0f));
  }

  //get gradient of shared bias
  mBiasCorrection.Scale(mMomentum);
  for(int r=0; r<E.Rows(); r++) {
    for(int c=0; c<E.Cols(); c++) {
      mBiasCorrection[c%mBiasCorrection.Dim()] += E(r,c);
    }
  }

  //perform update 
  mLinearity.AddScaled(-mLearningRate/N,mLinearityCorrection);
  mBias.AddScaled(-mLearningRate/N,mBiasCorrection);
  
  //regularization weight decay
  mLinearity.AddScaled(-mLearningRate*mWeightcost,mLinearity);
}
#endif

void
SharedLinearity::
ReadFromStream(std::istream& rIn)
{
  //number of instances of shared weights in layer
  rIn >> std::ws >> mNInstances;
  if(mNInstances < 1) {
    std::ostringstream os;
    os << "Bad number of instances:" << mNInstances;
    Error(os.str());
  }
  if(GetNInputs() % mNInstances != 0 || GetNOutputs() % mNInstances != 0) {
    std::ostringstream os;
    os << "Number of Inputs/Outputs must be divisible by number of instances"
       << " Inputs:" << GetNInputs()
       << " Outputs" << GetNOutputs()
       << " Intances:" << mNInstances;
    Error(os.str());
  }
    
  //matrix is stored transposed as SNet does
  BfMatrix transpose;
  rIn >> transpose;
  mLinearity = BfMatrix(transpose, TRANS);
  //biases stored normally
  rIn >> mBias;

  if(transpose.Cols()*transpose.Rows() == 0) {
    Error("Missing linearity matrix in network file");
  }
  if(mBias.Dim() == 0) {
    Error("Missing bias vector in network file");
  }


  if(mLinearity.Cols() != (GetNOutputs() / mNInstances) || 
     mLinearity.Rows() != (GetNInputs() / mNInstances) ||
     mBias.Dim() != (GetNOutputs() / mNInstances)
  ){
    std::ostringstream os;
    os << "Wrong dimensionalities of matrix/vector in network file\n"
       << "Inputs:" << GetNInputs()
       << " Outputs:" << GetNOutputs()
       << "\n"
       << "N-Instances:" << mNInstances
       << "\n"
       << "linearityCols:" << mLinearity.Cols() << "(" << mLinearity.Cols()*mNInstances << ")"
       << " linearityRows:" << mLinearity.Rows() << "(" << mLinearity.Rows()*mNInstances << ")"
       << " biasDims:" << mBias.Dim() << "(" << mBias.Dim()*mNInstances << ")"
       << "\n";
    Error(os.str());
  }

  mLinearityCorrection.Init(mLinearity.Rows(),mLinearity.Cols());
  mBiasCorrection.Init(mBias.Dim());
}

 
void
SharedLinearity::
WriteToStream(std::ostream& rOut)
{
  rOut << mNInstances << std::endl;
  //matrix is stored transposed as SNet does
  BfMatrix transpose(mLinearity, TRANS);
  rOut << transpose;
  //biases stored normally
  rOut << mBias;
  rOut << std::endl;
}


void 
SharedLinearity::
Gradient() 
{
  const Matrix<BaseFloat>& X = GetInput();
  const Matrix<BaseFloat>& E = GetErrorInput();
  //get gradient of shared linearity
  for(int k=0; k<mNInstances; k++) {
    SubMatrix<BaseFloat> xblock(X,0,X.Rows(),k*mpLinearity->Rows(),mpLinearity->Rows());
    SubMatrix<BaseFloat> eblock(E,0,E.Rows(),k*mpLinearity->Cols(),mpLinearity->Cols());
    mLinearityCorrection.BlasGemm(1.0,xblock,TRANS,eblock,NO_TRANS,((k==0)?0.0f:1.0f));
  }

  //get gradient of shared bias
  mBiasCorrection.Set(0.0f);
  for(int r=0; r<E.Rows(); r++) {
    for(int c=0; c<E.Cols(); c++) {
      mBiasCorrection[c%mBiasCorrection.Dim()] += E(r,c);
    }
  }
}


void 
SharedLinearity::
AccuGradient(const UpdatableComponent& src, int thr, int thrN)
{
  //cast the argument
  const SharedLinearity& src_comp = dynamic_cast<const SharedLinearity&>(src);

  //allocate accumulators when needed
  if(mLinearityCorrectionAccu.MSize() == 0) {
    mLinearityCorrectionAccu.Init(mpLinearity->Rows(),mpLinearity->Cols());
  }
  if(mBiasCorrectionAccu.MSize() == 0) {
    mBiasCorrectionAccu.Init(mpBias->Dim());
  }
 

  //assert the dimensions
  /*
  assert(mLinearityCorrection.Rows() == src_comp.mLinearityCorrection.Rows());
  assert(mLinearityCorrection.Cols() == src_comp.mLinearityCorrection.Cols());
  assert(mBiasCorrection.Dim() == src_comp.mBiasCorrection.Dim());
  */

  //need to find out which rows to sum...
  int div = mLinearityCorrection.Rows() / thrN;
  int mod = mLinearityCorrection.Rows() % thrN;

  int origin = thr * div + ((mod > thr)? thr : mod);
  int rows = div + ((mod > thr)? 1 : 0);

  //std::cout << "[S" << thr << "," << origin << "," << rows << "]" << std::flush;

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

  //first thread will always sum the bias correction and adds frame count
  if(thr == 0) {
    //std::cout << "[BS" << thr << "]" << std::flush;
    Add(mBiasCorrectionAccu,src_comp.mBiasCorrection);
  }
}


void 
SharedLinearity::
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

  //TODO perform L2 regularization
  //tgt_mat.AddScaled(tgt_mat, -mWeightcost * num_frames);

  //update weights
  AddScaled(tgt_mat, src_mat, -mLearningRate/static_cast<BaseFloat>(mNInstances));

  //first thread always update bias
  if(thr == 0) {
    //std::cout << "[" << thr << "BP]" << std::flush;
    AddScaled(mBias, mBiasCorrectionAccu, -mLearningRate/static_cast<BaseFloat>(mNInstances));
  }

  //reset the accumulators
  src_mat.Zero();
  if(thr == 0) {
    mBiasCorrectionAccu.Zero();
  }
}

 
} //namespace
