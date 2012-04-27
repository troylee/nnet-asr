

#include "BlockArray.h"
#include "Nnet.h"


namespace TNet
{

  void 
  BlockArray::
  PropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y)
  {
    SubMatrix<BaseFloat> colsX(X,0,1,0,1); //dummy dimensions
    SubMatrix<BaseFloat> colsY(Y,0,1,0,1); //dummy dimensions
    
    int X_src_ori=0, Y_tgt_ori=0;
    for(int i=0; i<mNBlocks; i++) {
      //get the correct submatrices
      int colsX_cnt=mBlocks[i]->GetNInputs();
      int colsY_cnt=mBlocks[i]->GetNOutputs();
      colsX = X.Range(0,X.Rows(),X_src_ori,colsX_cnt);
      colsY = Y.Range(0,Y.Rows(),Y_tgt_ori,colsY_cnt);

      //propagate through the block(network)
      mBlocks[i]->Propagate(colsX,colsY);

      //shift the origin coordinates
      X_src_ori += colsX_cnt;
      Y_tgt_ori += colsY_cnt;
    }

    assert(X_src_ori == X.Cols());
    assert(Y_tgt_ori == Y.Cols());
  }


  void 
  BlockArray::
  BackpropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y)
  {
    KALDI_ERR << "Unimplemented";
  }

  
  void 
  BlockArray::
  Update() 
  {
    KALDI_ERR << "Unimplemented";
  }


  void
  BlockArray::
  ReadFromStream(std::istream& rIn)
  {
    if(mBlocks.size() > 0) {
      KALDI_ERR << "Cannot read block vector, "
                << "aleady filled bt "
                << mBlocks.size()
                << "elements";
    }

    rIn >> std::ws >> mNBlocks;
    if(mNBlocks < 1) {
      KALDI_ERR << "Bad number of blocks:" << mNBlocks;
    }

    //read all the blocks
    std::string tag;
    int block_id;
    for(int i=0; i<mNBlocks; i++) {
      //read tag <block>
      rIn >> std::ws >> tag;
      //make it lowercase
      std::transform(tag.begin(), tag.end(), tag.begin(), tolower);
      //check
      if(tag!="<block>") {
        KALDI_ERR << "<block> keywotd expected";
      }
    
      //read block number
      rIn >> std::ws >> block_id;
      if(block_id != i+1) {
        KALDI_ERR << "Expected block number:" << i+1
                  << " read block number: " << block_id;
      }

      //read the nnet
      Network* p_nnet = new Network;
      p_nnet->ReadNetwork(rIn);
      if(p_nnet->Layers() == 0) {
        KALDI_ERR << "Cannot read empty network to a block";
      }

      //add it to the vector
      mBlocks.push_back(p_nnet);
    }

    //check the declared dimensionality
    int sum_inputs=0, sum_outputs=0;
    for(int i=0; i<mNBlocks; i++) {
      sum_inputs += mBlocks[i]->GetNInputs();
      sum_outputs += mBlocks[i]->GetNOutputs();
    }
    if(sum_inputs != GetNInputs()) {
      KALDI_ERR << "Non-matching number of INPUTS! Declared:"
                << GetNInputs()
                << " summed from blocks"
                << sum_inputs;
    }
    if(sum_outputs != GetNOutputs()) {
      KALDI_ERR << "Non-matching number of OUTPUTS! Declared:"
                << GetNOutputs()
                << " summed from blocks"
                << sum_outputs;
    }
  }

   
  void
  BlockArray::
  WriteToStream(std::ostream& rOut)
  {
    rOut << " " << mBlocks.size() << " ";
    for(int i=0; i<mBlocks.size(); i++) {
      rOut << "<block> " << i+1 << "\n";
      mBlocks[i]->WriteNetwork(rOut);
      rOut << "<endblock>\n";
    }
  }

 
} //namespace

