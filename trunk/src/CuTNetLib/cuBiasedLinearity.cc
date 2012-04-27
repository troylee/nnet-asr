

#include "cuBiasedLinearity.h"


namespace TNet
{

  void 
  CuBiasedLinearity::
  PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    //Y.SetConst(0.0);
    Y.AddScaledRow(1.0,mBias,0.0);
    Y.Gemm('N','N', 1.0, X, mLinearity, 1.0);
  }


  void 
  CuBiasedLinearity::
  BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    //Y.SetConst(0.0);
    Y.Gemm('N', 'T', 1.0, X, mLinearity, 0.0);
  }

  
  void 
  CuBiasedLinearity::
  Update() 
  {
#if 0
    //former implementation
    BaseFloat N = static_cast<BaseFloat>(GetInput().Rows());

    mLinearityCorrection.Gemm('T','N',-mLearningRate/N,GetInput(),GetErrorInput(),mMomentum);
    mBiasCorrection.AddColSum(-mLearningRate/N,GetErrorInput(),mMomentum);

    //regularization weight decay
    mLinearityCorrection.AddScaled(-mLearningRate*mWeightcost,mLinearity,1.0);
    
    mLinearity.AddScaled(1.0,mLinearityCorrection,1.0);
    mBias.AddScaled(1.0,mBiasCorrection,1.0);
#endif

#if 1
    //new implementation
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

    //regularization weight decay (from actual weights only)
    BaseFloat L2_decay = -mLearningRate*mWeightcost*(mGradDivFrm?1.0:GetInput().Rows());
    mLinearity.AddScaled(L2_decay, mLinearity,1.0);
#endif
  }


  void
  CuBiasedLinearity::
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
  }

   
  void
  CuBiasedLinearity::
  WriteToStream(std::ostream& rOut)
  {
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
  }

 
} //namespace

