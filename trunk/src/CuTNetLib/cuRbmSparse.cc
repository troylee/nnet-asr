
#include <string>
#include <sstream>

#include "cuRbmSparse.h"

#include "cumath.h"


namespace TNet
{

  void 
  CuRbmSparse::
  PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    Y.SetConst(0.0);
    Y.AddScaledRow(1.0,mHidBias,0.0);
    Y.Gemm('N','N', 1.0, X, mVisHid, 1.0);
    if(mHidType == BERNOULLI) {
      CuMath<BaseFloat>::Sigmoid(Y,Y);
    }
  }


  void 
  CuRbmSparse::
  BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    if(mHidType == BERNOULLI) {
      mBackpropErrBuf.Init(X.Rows(),X.Cols());
      CuMath<BaseFloat>::DiffSigmoid(mBackpropErrBuf,X,GetOutput());
    } else {
      mBackpropErrBuf.CopyFrom(X);
    }

    Y.SetConst(0.0);
    Y.Gemm('N', 'T', 1.0, mBackpropErrBuf, mVisHid, 0.0);
  }

  
  void 
  CuRbmSparse::
  Update() 
  {
    //THIS IS DONE TWICE BECAUSE OF THE BACKPROP STOPPER!!!
    if(mHidType == BERNOULLI) {
      mBackpropErrBuf.Init(GetErrorInput().Rows(),GetErrorInput().Cols());
      CuMath<BaseFloat>::DiffSigmoid(mBackpropErrBuf,GetErrorInput(),GetOutput());
    } else {
      mBackpropErrBuf.CopyFrom(GetErrorInput());
    }

/*
    std::cout << " " << GetInput().Rows()
              << " " << GetInput().Cols()  
              << " " << mBackpropErrBuf.Rows()  
              << " " << mBackpropErrBuf.Cols()  
              << " " << mVisHidCorrection.Rows()  
              << " " << mVisHidCorrection.Cols()  
              ;
*/

#if 0
    //former implementation
    BaseFloat N = static_cast<BaseFloat>(GetInput().Rows());

    mVisHidCorrection.Gemm('T','N',-mLearningRate/N,GetInput(),mBackpropErrBuf,mMomentum);
    mHidBiasCorrection.AddColSum(-mLearningRate/N,mBackpropErrBuf,mMomentum);

    //regularization weight decay
    mVisHidCorrection.AddScaled(-mLearningRate*mWeightcost,mVisHid,1.0);
    
    mVisHid.AddScaled(1.0,mVisHidCorrection,1.0);
    mHidBias.AddScaled(1.0,mHidBiasCorrection,1.0);
#endif

#if 1
    //new implementation
    BaseFloat N = 1;
    if(mGradDivFrm) {
      N = static_cast<BaseFloat>(GetInput().Rows());
    }
    BaseFloat mmt_gain = static_cast<BaseFloat>(1.0/(1.0-mMomentum));
    N *= mmt_gain;

    mVisHidCorrection.Gemm('T','N',1.0,GetInput(),mBackpropErrBuf,mMomentum);
    mHidBiasCorrection.AddColSum(1.0,mBackpropErrBuf,mMomentum);

    mVisHid.AddScaled(-mLearningRate/N,mVisHidCorrection,1.0);
    mHidBias.AddScaled(-mLearningRate/N,mHidBiasCorrection,1.0);

    //regularization weight decay (from actual weights only)
    mVisHid.AddScaled(-mLearningRate*mWeightcost,mVisHid,1.0);
#endif

  }



  void 
  CuRbmSparse::
  Propagate(const CuMatrix<BaseFloat>& visProbs, CuMatrix<BaseFloat>& hidProbs)
  {
    if(visProbs.Cols() != GetNInputs()) {
      std::ostringstream os;
      os << " Nonmatching input dim, needs:" << GetNInputs() 
         << " got:" << visProbs.Cols() << "\n";
      Error(os.str());
    }

    hidProbs.Init(visProbs.Rows(),GetNOutputs());

    PropagateFnc(visProbs, hidProbs);
  }

  void
  CuRbmSparse::
  Reconstruct(const CuMatrix<BaseFloat>& hidState, CuMatrix<BaseFloat>& visProbs)
  {
    visProbs.Init(hidState.Rows(),mNInputs);
    visProbs.SetConst(0.0);
    visProbs.AddScaledRow(1.0,mVisBias,0.0);
    visProbs.Gemm('N','T', 1.0, hidState, mVisHid, 1.0);
    if(mVisType == BERNOULLI) {
      CuMath<BaseFloat>::Sigmoid(visProbs,visProbs);
    }
  }


  void 
  CuRbmSparse::
  RbmUpdate(const CuMatrix<BaseFloat>& pos_vis, const CuMatrix<BaseFloat>& pos_hid, const CuMatrix<BaseFloat>& neg_vis, const CuMatrix<BaseFloat>& neg_hid)
  {
    assert(pos_vis.Rows() == pos_hid.Rows() &&
           pos_vis.Rows() == neg_vis.Rows() &&
           pos_vis.Rows() == neg_hid.Rows() &&
           pos_vis.Cols() == neg_vis.Cols() &&
           pos_hid.Cols() == neg_hid.Cols() &&
           pos_vis.Cols() == mNInputs &&
           pos_hid.Cols() == mNOutputs);

    //:SPARSITY:
    if(mHidType==BERNOULLI) {
      //get expected node activity from current batch
      mSparsityQCurrent.AddColSum(1.0/pos_hid.Rows(),pos_hid,0.0);
      //get smoothed expected node activity
      mSparsityQ.AddScaled(1.0-mLambda,mSparsityQCurrent,mLambda);
      //subtract the prior: (q-p)
      mSparsityQCurrent.SetConst(-mSparsityPrior);
      mSparsityQCurrent.AddScaled(1.0,mSparsityQ,1.0);
      //get mean pos_vis
      mVisMean.AddColSum(1.0/pos_vis.Rows(),pos_vis,0.0);
    }

    //  UPDATE vishid matrix
    //  
    //  vishidinc = momentum*vishidinc + ...
    //              epsilonw*( (posprods-negprods)/numcases - weightcost*vishid)
    //              -sparsitycost*mean_posvis'*(q-p);
    //
    //  vishidinc[t] = -(epsilonw/numcases)*negprods + momentum*vishidinc[t-1]
    //                 +(epsilonw/numcases)*posprods
    //                 -(epsilonw*weightcost)*vishid[t-1]
    //
    BaseFloat N = static_cast<BaseFloat>(pos_vis.Rows());
    mVisHidCorrection.Gemm('T','N',-mLearningRate/N,neg_vis,neg_hid,mMomentum);
    mVisHidCorrection.Gemm('T','N',+mLearningRate/N,pos_vis,pos_hid,1.0);
    mVisHidCorrection.AddScaled(-mLearningRate*mWeightcost,mVisHid,1.0);//L2
    if(mHidType==BERNOULLI) {
      mVisHidCorrection.BlasGer(-mSparsityCost,mVisMean,mSparsityQCurrent);//sparsity
    }
    mVisHid.AddScaled(1.0,mVisHidCorrection,1.0);

    //  UPDATE visbias vector
    //
    //  visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
    //
    mVisBiasCorrection.AddColSum(-mLearningRate/N,neg_vis,mMomentum);
    mVisBiasCorrection.AddColSum(+mLearningRate/N,pos_vis,1.0);
    mVisBias.AddScaled(1.0,mVisBiasCorrection,1.0);
    
    //  UPDATE hidbias vector
    //
    // hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);
    //
    mHidBiasCorrection.AddColSum(-mLearningRate/N,neg_hid,mMomentum);
    mHidBiasCorrection.AddColSum(+mLearningRate/N,pos_hid,1.0);
    if(mHidType==BERNOULLI) {
      mHidBiasCorrection.AddScaled(-mSparsityCost,mSparsityQCurrent,1.0);//sparsity
    }
    mHidBias.AddScaled(1.0/*0.0*/,mHidBiasCorrection,1.0);

  }


  void
  CuRbmSparse::
  ReadFromStream(std::istream& rIn)
  {
    //type of the units
    std::string str;
    
    rIn >> std::ws >> str;
    if(0 == str.compare("bern")) {
      mVisType = BERNOULLI;
    } else if(0 == str.compare("gauss")) {
      mVisType = GAUSSIAN;
    } else Error(std::string("Invalid unit type: ")+str);

    rIn >> std::ws >> str;
    if(0 == str.compare("bern")) {
      mHidType = BERNOULLI;
    } else if(0 == str.compare("gauss")) {
      mHidType = GAUSSIAN;
    } else Error(std::string("Invalid unit type: ")+str);


    //matrix is stored transposed as SNet does
    BfMatrix transpose;
    rIn >> transpose;
    mVisHid.CopyFrom(BfMatrix(transpose, TRANS));
    //biases stored normally
    BfVector bias;
    rIn >> bias;
    mVisBias.CopyFrom(bias);
    rIn >> bias;
    mHidBias.CopyFrom(bias);

    rIn >> std::ws >> mSparsityCost;
    std::cout << "RBM::mSparsityCost=" << mSparsityCost;
  }

   
  void
  CuRbmSparse::
  WriteToStream(std::ostream& rOut)
  {
    //store unit type info
    if(mVisType == BERNOULLI) {
      rOut << " bern ";
    } else {
      rOut << " gauss ";
    }
    if(mHidType == BERNOULLI) {
      rOut << " bern\n";
    } else {
      rOut << " gauss\n";
    }

    //matrix is stored transposed as SNet does
    BfMatrix tmp;
    mVisHid.CopyTo(tmp);
    BfMatrix transpose(tmp, TRANS);
    rOut << transpose;
    //biases stored normally
    BfVector vec;
    mVisBias.CopyTo(vec);
    rOut << vec;
    rOut << std::endl;
    mHidBias.CopyTo(vec);
    rOut << vec;
    rOut << std::endl;
    //store the sparsity cost
    rOut << mSparsityCost << std::endl;
  }

 
} //namespace
