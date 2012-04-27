
#include <string>
#include <sstream>

#include "cuRecurrent.h"

#include "cumath.h"
#include "cuda_runtime.h"


namespace TNet
{

  void 
  CuRecurrent::
  PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    assert(X.Rows() == 1);
    assert(Y.Rows() == 1);
    if(mInputHistory.Rows() == 0) {
      Error("Bptt order was not set");
    }

    //pushback the history
    CuMatrix<BaseFloat> tmp(mInputHistory.Rows()-1,mInputHistory.Cols());
    tmp.CopyRows(tmp.Rows(),0,mInputHistory,0);
    mInputHistory.CopyRows(tmp.Rows(),0,tmp,1);

    //compose the input vector to 0th row, use input X and previous Y
    cudaMemcpy(mInputHistory.pCUData(), X.pCUData(),
               sizeof(BaseFloat)*X.Cols(), cudaMemcpyDeviceToDevice);
    cudaMemcpy(mInputHistory.pCUData()+X.Cols(), Y.pCUData(),
               sizeof(BaseFloat)*Y.Cols(), cudaMemcpyDeviceToDevice);

    //extract first row
    //CuMatrix<BaseFloat> first_row(1,mInputHistory.Cols());
    //first_row.CopyRows(1,0,mInputHistory,0);

    //calculate the output
    Y.AddScaledRow(1.0,mBias,0.0);
    //take 0th vector of history, propagate
    CuMath<BaseFloat>::OffsetGemv('T',1.0,mLinearity,mInputHistory.pCUData(),mInputHistory.Cols(),1.0,Y.pCUData(),Y.Cols(),0); 
    //Y.Gemm('N','N', 1.0, first_row, mLinearity, 1.0);
    CuMath<BaseFloat>::Sigmoid(Y,Y);

    /*
    std::cout << "-------------------------------------" << std::endl;
    X.Print();
    Y.Print();
    mInputHistory.Print();
    */

  }


  void 
  CuRecurrent::
  BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    assert(Y.Rows() == 1);
    assert(X.Rows() == 1);

    //apply diff sigmoid
    CuMatrix<BaseFloat> diff_sigm(1,X.Cols());
    CuMath<BaseFloat>::DiffSigmoid(diff_sigm,X,GetOutput());
    
    //:TODO: inefficent to calculate all the input errors!!!
    //       we need only part of them!
    //
    //backward-multiply by weights
    /*
    CuMatrix<BaseFloat> err_prev(1,mLinearity.Rows());
    err_prev.Gemm('N', 'T', 1.0, diff_sigm, mLinearity, 0.0);
 
    //copy out the interval
    cudaMemcpy(Y.pCUData(),err_prev.pCUData(),
               sizeof(BaseFloat)*Y.Cols(),cudaMemcpyDeviceToDevice);
    */

    //backward-multiply by weights
    CuMath<BaseFloat>::OffsetGemv('N',1.0,mLinearity,diff_sigm.pCUData(),diff_sigm.Cols(),1.0,Y.pCUData(),Y.Cols(),0); 

  }

  
  void 
  CuRecurrent::
  Update() 
  {
    //
    //correction from PRESENT input x error pair
    //
    //apply diff sigmoid
    CuMatrix<BaseFloat> diff_sigm(1,GetOutput().Cols());
    CuMath<BaseFloat>::DiffSigmoid(diff_sigm,GetErrorInput(),GetOutput());

    //get 0th row of history (present time)
    CuMatrix<BaseFloat> history_row(1,mInputHistory.Cols());
    history_row.CopyRows(1,0,mInputHistory,0);

    //calculate update
    //mLinearityCorrection.Gemm('T','N',-mLearningRate,history_row,diff_sigm,mMomentum);
    mLinearityCorrection.SetConst(0.0); //:TODO: should be scale/momentum
    CuMath<BaseFloat>::BlasGer(-mLearningRate,history_row.pCUData(),history_row.Cols(),diff_sigm.pCUData(),diff_sigm.Cols(),mLinearityCorrection);

    mBiasCorrection.AddColSum(-mLearningRate,diff_sigm,mMomentum);
   
    //
    //BPTT (backprop through time) 
    //
    CuMatrix<BaseFloat> err_prev(1,mLinearity.Rows());
    CuMatrix<BaseFloat> err_prev_part(1,diff_sigm.Cols());
    CuMatrix<BaseFloat> history_output(1,GetOutput().Cols());
    for(int i=1; i<=mBpttOrder; i++) {
      //:TODO: inefficent to calculate all the input errors!!!
      //       we need only part of them!
      //
      /*
      //get previous error
      err_prev.Gemm('N','T',1.0,diff_sigm,mLinearity,0.0);
      //select interval
      cudaMemcpy(err_prev_part.pCUData(),err_prev.pCUData()+GetNInputs(),
                 sizeof(BaseFloat)*err_prev_part.Cols(),cudaMemcpyDeviceToDevice);
      */

      //backward-multiply by weights
      CuMath<BaseFloat>::OffsetGemv('N',1.0,mLinearity,diff_sigm.pCUData(),diff_sigm.Cols(),0.0,err_prev_part.pCUData(),err_prev_part.Cols(),GetInput().Cols()); 

      //apply diff sigmoid with activations of HISTORY frame!!!
      cudaMemcpy(history_output.pCUData(), mInputHistory.pCURowData(i-1)+GetInput().Cols(),
          sizeof(BaseFloat)*history_output.Cols(), cudaMemcpyDeviceToDevice);
      CuMath<BaseFloat>::DiffSigmoid(diff_sigm,err_prev_part,history_output);

      //get history row
      history_row.CopyRows(1,i,mInputHistory,0);

      //accu the update
      //mLinearityCorrection.Gemm('T','N',-mLearningRate,history_row,diff_sigm,1.0);
      CuMath<BaseFloat>::BlasGer(-mLearningRate,history_row.pCUData(),history_row.Cols(),diff_sigm.pCUData(),diff_sigm.Cols(),mLinearityCorrection);
      mBiasCorrection.AddColSum(-mLearningRate,diff_sigm,1.0);
    }

    //
    //update the weights
    //
    //regularization weight decay
    mLinearityCorrection.AddScaled(-mLearningRate*mWeightcost,mLinearity,1.0);
    
    //perform update
    mLinearity.AddScaled(1.0,mLinearityCorrection,1.0);
    mBias.AddScaled(1.0,mBiasCorrection,1.0);

  }




  void
  CuRecurrent::
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
  }

   
  void
  CuRecurrent::
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

