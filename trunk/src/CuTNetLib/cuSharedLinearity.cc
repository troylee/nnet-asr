

#include "cuSharedLinearity.h"
#include "cumath.h"


namespace TNet
{

  void 
  CuSharedLinearity::
  PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    CuMath<BaseFloat>::VecExpand(mBias,mBiasExpand); /// [ 1 2 3 ] -> [ 1 2 3 1 2 3 ... ]
    Y.AddScaledRow(1.0,mBiasExpand,0.0);

    //mBiasExpand.Print();

    for(int i=0; i<mNInstances; i++) {
      CuMath<BaseFloat>::OffsetGemm('N','N', 1.0, X, mLinearity, 1.0, Y, 
                                    i*mLinearity.Rows(), 0, i*mLinearity.Cols());
    }
    //std::cout << CuDevice::Instantiate().GetFreeMemory();
    //GetInput().Print();
    //GetOutput().Print();
  }


  void 
  CuSharedLinearity::
  BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    for(int i=0; i<mNInstances; i++) {
      CuMath<BaseFloat>::OffsetGemm('N', 'T', 1.0, X, mLinearity, 0.0, Y,
                                    i*mLinearity.Cols(), 0, i*mLinearity.Rows());
    }
  }

  
  void 
  CuSharedLinearity::
  Update() 
  {
#if 0
    //former implementation
    BaseFloat N = static_cast<BaseFloat>(GetInput().Rows());

    for(int i=0; i<mNInstances; i++) {
      CuMath<BaseFloat>::OffsetGemm('T','N',-mLearningRate/(N*mNInstances),
                        GetInput(),GetErrorInput(),
                        ((i==0)?mMomentum:1.0f), mLinearityCorrection, 
                        i*mLinearity.Rows(),i*mLinearity.Cols(),0);
    }
    mBiasCorrectionExpand.AddColSum(1.0,GetErrorInput(),0.0);
    CuMath<BaseFloat>::VecAddColSum(-mLearningRate/(N*mNInstances),mBiasCorrectionExpand,mMomentum,mBiasCorrection);


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
    N *= mmt_gain; //compensate higher gradient estimates due to momentum 
    
    //compensate augmented dyn. range of gradient caused by multiple instances
    N *= static_cast<BaseFloat>(mNInstances); 

    //get gradient of shared linearity
    for(int i=0; i<mNInstances; i++) {
      CuMath<BaseFloat>::OffsetGemm('T','N',1.0,
                        GetInput(),GetErrorInput(),
                        ((i==0)?mMomentum:1.0f), mLinearityCorrection, 
                        i*mLinearity.Rows(),i*mLinearity.Cols(),0);
    }
    //get gradient of shared bias
    mBiasCorrectionExpand.AddColSum(1.0,GetErrorInput(),0.0);
    CuMath<BaseFloat>::VecAddColSum(1.0,mBiasCorrectionExpand,mMomentum,mBiasCorrection);
   
    //perform update 
    mLinearity.AddScaled(-mLearningRate/N,mLinearityCorrection,1.0);
    mBias.AddScaled(-mLearningRate/N,mBiasCorrection,1.0);
    
    //regularization weight decay
    mLinearity.AddScaled(-mLearningRate*mWeightcost,mLinearity,1.0);
#endif
   
  }


  void
  CuSharedLinearity::
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


    if(mLinearity.Cols() != GetNOutputs() / mNInstances || 
       mLinearity.Rows() != GetNInputs() / mNInstances ||
       mBias.Dim() != GetNOutputs() / mNInstances
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

    mLinearityCorrection.Init(mLinearity.Rows(),mLinearity.Cols());
    mBiasCorrection.Init(mBias.Dim());

    mBiasExpand.Init(mBias.Dim()*mNInstances);
    mBiasCorrectionExpand.Init(mBias.Dim()*mNInstances);
  }

   
  void
  CuSharedLinearity::
  WriteToStream(std::ostream& rOut)
  {
    rOut << mNInstances << std::endl;

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
