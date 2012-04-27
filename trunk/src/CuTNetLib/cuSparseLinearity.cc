

#include "cuSparseLinearity.h"
#include <cmath>
#include <cstdlib>


namespace TNet
{

  void 
  CuSparseLinearity::
  PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    Y.AddScaledRow(1.0,mBias,0.0);
    Y.Gemm('N','N', 1.0, X, mLinearity, 1.0);
  }


  void 
  CuSparseLinearity::
  BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    Y.Gemm('N', 'T', 1.0, X, mLinearity, 0.0);
  }

  
  void 
  CuSparseLinearity::
  Update() 
  {
    BaseFloat N = 1;
    if(mGradDivFrm) {
      N = static_cast<BaseFloat>(GetInput().Rows());
    }
    BaseFloat mmt_gain = static_cast<BaseFloat>(1.0/(1.0-mMomentum));
    N *= mmt_gain;

    mLinearityCorrection.Gemm('T','N',1.0,GetInput(),GetErrorInput(),mMomentum);
    mBiasCorrection.AddColSum(1.0,GetErrorInput(),mMomentum);

    mLinearity.AddScaled(-mLearningRate/N,mLinearityCorrection,1.0);
    mBias.AddScaled(-mLearningRate/N,mBiasCorrection,1.0);

    mLinearityCorrectionAccu.AddScaled(1.0,mLinearityCorrection,1.0);
    mLinearity.ApplyMask(mSparsityMask); 

    //L1 regularization lassoo...
    //each update? everty 1000th update?
    if(mL1Const > 0) {
      BaseFloat L1_const = mLearningRate*mL1Const*(mGradDivFrm?1.0:GetInput().Rows());
      mLinearity.ApplyL1(L1_const);
    }

    //L2 regularization weight decay (from actual weights only)
    if(mWeightcost > 0) {
      BaseFloat L2_decay = -mLearningRate*mWeightcost*(mGradDivFrm?1.0:GetInput().Rows());
      mLinearity.AddScaled(L2_decay, mLinearity,1.0);
    }

    mNFrames += GetInput().Rows();

  }


  void 
  CuSparseLinearity::
  UpdateMask()
  {
    //move data to host
    Matrix<BaseFloat> linearity, linearity_correction_accu; 
    Matrix<BaseFloat> sparsity_mask;

    mLinearity.CopyTo(linearity);
    mLinearityCorrectionAccu.CopyTo(linearity_correction_accu);
    mSparsityMask.CopyTo(sparsity_mask);

    //decide on new sparsity mask
    for(size_t r=0; r<sparsity_mask.Rows(); r++) {
      for(size_t c=0; c<sparsity_mask.Cols(); c++) {
        if(sparsity_mask(r,c) == 1.0f) { //weight active
          if(fabs(linearity(r,c)) < mSparsifyWeightThreshold) {
            sparsity_mask(r,c) = 0;//deactivate
            linearity(r,c) = 0;
          }
        } else { //weight inactive
          if(abs(linearity_correction_accu(r,c))/(BaseFloat)mNFrames > mUnsparsifyAccu) {
            sparsity_mask(r,c) = 1;//activate
          }
        }
      }
    }

    //move data to the device
    mLinearity.CopyFrom(linearity);
    mSparsityMask.CopyFrom(sparsity_mask);
  }


  void
  CuSparseLinearity::
  ReadFromStream(std::istream& rIn)
  {
    //matrix is stored transposed as SNet does
    BfMatrix transpose;
    rIn >> transpose;
    mLinearity.CopyFrom(BfMatrix(transpose, TRANS));
    //biases stored normally
    BfVector bias;
    rIn >> bias;
    mBias.CopyFrom(bias);

    //sparsity mask
    rIn >> std::ws;
    Matrix<BaseFloat> mask_transp;
    if(rIn.peek() == 'm') {//load from file
      rIn >> mask_transp;
    } else {//or set all elements active
      mask_transp.Init(transpose.Rows(),transpose.Cols());
      int items=transpose.Rows()*transpose.Stride();
      BaseFloat* p = mask_transp.pData();
      for(int i=0; i<items; i++) {//set all elements to one
        *p++ = 1;
      }
    }
    mSparsityMask.CopyFrom(BfMatrix(mask_transp,TRANS));

    //dummy matrix with acumulated gradients
    rIn >> std::ws;
    if(rIn.peek() == 'm') {//load from file
      BfMatrix dummy;
      rIn >> dummy;
    }

    if(transpose.Cols()*transpose.Rows() == 0) {
      Error("Missing linearity matrix in network file");
    }
    if(bias.Dim() == 0) {
      Error("Missing bias vector in network file");
    }
    if(mLinearity.Cols() != GetNOutputs() || 
       mLinearity.Rows() != GetNInputs() ||
       mBias.Dim() != GetNOutputs()
    ){
      std::ostringstream os;
      os << "Wrong dimensionalities of matrix/vector in network file\n"
         << "Inputs:" << GetNInputs()
         << "Outputs:" << GetNOutputs()
         << "\n"
         << "linearityCols:" << mLinearity.Cols()
         << "linearityRows:" << mLinearity.Rows()
         << "biasDims:" << mBias.Dim()
         << "\n";
      Error(os.str());
    }

    assert(mLinearity.Rows() == mSparsityMask.Rows());
    assert(mLinearity.Cols() == mSparsityMask.Cols());

  }

   
  void
  CuSparseLinearity::
  WriteToStream(std::ostream& rOut)
  {
    UpdateMask();

    //matrix is stored transposed as SNet does
    BfMatrix tmp;
    mLinearity.CopyTo(tmp);
    BfMatrix transpose(tmp, TRANS);
    rOut << transpose;
    //biases stored normally
    BfVector vec;
    mBias.CopyTo(vec);
    rOut << vec;
    rOut << std::endl;
    //store mask
    mSparsityMask.CopyTo(tmp);
    rOut << BfMatrix(tmp,TRANS);
    //store accu
    mLinearityCorrectionAccu.CopyTo(tmp);
    rOut << BfMatrix(tmp,TRANS);

  }

 
} //namespace

