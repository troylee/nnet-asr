
#include <algorithm>
//#include <locale>
#include <cctype>

#include "Nnet.h"
#include "CRBEDctFeat.h"
#include "BlockArray.h"

namespace TNet {




void Network::Feedforward(const Matrix<BaseFloat>& in, Matrix<BaseFloat>& out, 
                          size_t start_frm_ext, size_t end_frm_ext) {
  //empty network: copy input to output 
  if(mNnet.size() == 0) {
    if(out.Rows() != in.Rows() || out.Cols() != in.Cols()) {
      out.Init(in.Rows(),in.Cols());
    }
    out.Copy(in);
    return;
  }
  
  //short input: propagate in one block  
  if(in.Rows() < 5000) { 
    Propagate(in,out);
  } else {//long input: propagate per parts
    //initialize
    out.Init(in.Rows(),GetNOutputs());
    Matrix<BaseFloat> tmp_in, tmp_out;
    int done=0, block=1024;
    //propagate first part
    tmp_in.Init(block+end_frm_ext,in.Cols());
    tmp_in.Copy(in.Range(0,block+end_frm_ext,0,in.Cols()));
    Propagate(tmp_in,tmp_out);
    out.Range(0,block,0,tmp_out.Cols()).Copy(
      tmp_out.Range(0,block,0,tmp_out.Cols())
    );
    done += block;
    //propagate middle parts
    while((done+2*block) < in.Rows()) {
      tmp_in.Init(block+start_frm_ext+end_frm_ext,in.Cols());
      tmp_in.Copy(in.Range(done-start_frm_ext, block+start_frm_ext+end_frm_ext, 0,in.Cols()));      Propagate(tmp_in,tmp_out);
      out.Range(done,block,0,tmp_out.Cols()).Copy(
        tmp_out.Range(start_frm_ext,block,0,tmp_out.Cols())
      );
      done += block;
    }
    //propagate last part
    tmp_in.Init(in.Rows()-done+start_frm_ext,in.Cols());
    tmp_in.Copy(in.Range(done-start_frm_ext,in.Rows()-done+start_frm_ext,0,in.Cols()));
    Propagate(tmp_in,tmp_out);
    out.Range(done,out.Rows()-done,0,out.Cols()).Copy(
      tmp_out.Range(start_frm_ext,tmp_out.Rows()-start_frm_ext,0,tmp_out.Cols())   
    );

    done += tmp_out.Rows()-start_frm_ext;
    assert(done == out.Rows());
  }
}


void Network::Propagate(const Matrix<BaseFloat>& in, Matrix<BaseFloat>& out) {
  //empty network: copy input to output 
  if(mNnet.size() == 0) {
    if(out.Rows() != in.Rows() || out.Cols() != in.Cols()) {
      out.Init(in.Rows(),in.Cols());
    }
    out.Copy(in);
    return;
  }
  
  //this will keep pointer to matrix 'in', for backprop
  mNnet.front()->SetInput(in); 

  //propagate
  LayeredType::iterator it;
  for(it=mNnet.begin(); it!=mNnet.end(); ++it) {
    (*it)->Propagate();
  }

  //copy the output matrix
  const Matrix<BaseFloat>& mat = mNnet.back()->GetOutput();
  if(out.Rows() != mat.Rows() || out.Cols() != mat.Cols()) {
    out.Init(mat.Rows(),mat.Cols());
  }
  out.Copy(mat);

}


void Network::Backpropagate(const Matrix<BaseFloat>& globerr) {
  //pass matrix to last component
  mNnet.back()->SetErrorInput(globerr);

  // back-propagation : reversed order,
  LayeredType::reverse_iterator it;
  for(it=mNnet.rbegin(); it!=mNnet.rend(); ++it) {
    //first component does not backpropagate error (no predecessors)
    if(*it != mNnet.front()) {
      (*it)->Backpropagate();
    }
    //compute gradient if updatable component
    if((*it)->IsUpdatable()) {
      UpdatableComponent& comp = dynamic_cast<UpdatableComponent&>(**it);
      comp.Gradient(); //compute gradient 
    }
  }
}


void Network::AccuGradient(const Network& src, int thr, int thrN) {
  LayeredType::iterator it;
  LayeredType::const_iterator it2;

  for(it=mNnet.begin(), it2=src.mNnet.begin(); it!=mNnet.end(); ++it,++it2) {
    if((*it)->IsUpdatable()) {
      UpdatableComponent& comp = dynamic_cast<UpdatableComponent&>(**it);
      const UpdatableComponent& comp2 = dynamic_cast<const UpdatableComponent&>(**it2);
      comp.AccuGradient(comp2,thr,thrN);
    }
  }
}


void Network::Update(int thr, int thrN) {
  LayeredType::iterator it;

  for(it=mNnet.begin(); it!=mNnet.end(); ++it) {
    if((*it)->IsUpdatable()) {
      UpdatableComponent& comp = dynamic_cast<UpdatableComponent&>(**it);
      comp.Update(thr,thrN);
    }
  }
}


Network* Network::Clone() {
  Network* net = new Network;
  LayeredType::iterator it;
  for(it = mNnet.begin(); it != mNnet.end(); ++it) {
    //clone
    net->mNnet.push_back((*it)->Clone());
    //connect network
    if(net->mNnet.size() > 1) {
      Component* last = *(net->mNnet.end()-1);
      Component* prev = *(net->mNnet.end()-2);
      last->SetInput(prev->GetOutput());
      prev->SetErrorInput(last->GetErrorOutput());
    }
  }

  //copy the learning rate
  //net->SetLearnRate(GetLearnRate());

  return net;
}


void Network::ReadNetwork(const char* pSrc) {
  std::ifstream in(pSrc);
  if(!in.good()) {
    Error(std::string("Error, cannot read model: ")+pSrc);
  }
  ReadNetwork(in);
  in.close();
}

  

void Network::ReadNetwork(std::istream& rIn) {
  //get the network elements from a factory
  Component *pComp;
  while(NULL != (pComp = ComponentFactory(rIn))) 
    mNnet.push_back(pComp);
}


void Network::WriteNetwork(const char* pDst) {
  std::ofstream out(pDst);
  if(!out.good()) {
    Error(std::string("Error, cannot write model: ")+pDst);
  }
  WriteNetwork(out);
  out.close();
}


void Network::WriteNetwork(std::ostream& rOut) {
  //dump all the componetns
  LayeredType::iterator it;
  for(it=mNnet.begin(); it!=mNnet.end(); ++it) {
    ComponentDumper(rOut, **it);
  }
}
 

Component*
Network::
ComponentFactory(std::istream& rIn)
{
  rIn >> std::ws;
  if(rIn.eof()) return NULL;

  Component* pRet=NULL;
  Component* pPred=NULL;

  std::string componentTag;
  size_t nInputs, nOutputs;

  rIn >> std::ws;
  rIn >> componentTag;
  if(componentTag == "") return NULL; //nothing left in the file

  //make it lowercase
  std::transform(componentTag.begin(), componentTag.end(), 
                 componentTag.begin(), tolower);

  //the 'endblock' tag terminates the network
  if(componentTag == "<endblock>") return NULL;

  
  if(componentTag[0] != '<' || componentTag[componentTag.size()-1] != '>') {
    Error(std::string("Invalid component tag:")+componentTag);
  }

  rIn >> std::ws;
  rIn >> nOutputs;
  rIn >> std::ws;
  rIn >> nInputs;
  assert(nInputs > 0 && nOutputs > 0);

  //make coupling with predecessor
  if(mNnet.size() == 0) {
    pPred = NULL;
  } else {
    pPred = mNnet.back();
  }
  
  //array with list of component tags
  static const std::string TAGS[] = {
    "<biasedlinearity>",
    "<sharedlinearity>",
    
    "<sigmoid>",
    "<softmax>",
    "<blocksoftmax>",

    "<expand>",
    "<copy>",
    "<transpose>",
    "<blocklinearity>",
    "<bias>",
    "<window>",
    "<log>",

    "<blockarray>",
  };

  static const int n_tags = sizeof(TAGS) / sizeof(TAGS[0]);
  int i = 0;
  for(i=0; i<n_tags; i++) {
    if(componentTag == TAGS[i]) break;
  }
  
  //switch according to position in array TAGS
  switch(i) {
    case 0: pRet = new BiasedLinearity(nInputs,nOutputs,pPred); break;
    case 1: pRet = new SharedLinearity(nInputs,nOutputs,pPred); break;

    case 2: pRet = new Sigmoid(nInputs,nOutputs,pPred); break;
    case 3: pRet = new Softmax(nInputs,nOutputs,pPred); break;
    case 4: pRet = new BlockSoftmax(nInputs,nOutputs,pPred); break;

    case 5: pRet = new Expand(nInputs,nOutputs,pPred); break;
    case 6: pRet = new Copy(nInputs,nOutputs,pPred); break;
    case 7: pRet = new Transpose(nInputs,nOutputs,pPred); break;
    case 8: pRet = new BlockLinearity(nInputs,nOutputs,pPred); break;
    case 9: pRet = new Bias(nInputs,nOutputs,pPred); break;
    case 10: pRet = new Window(nInputs,nOutputs,pPred); break;
    case 11: pRet = new Log(nInputs,nOutputs,pPred); break;
    
    case 12: pRet = new BlockArray(nInputs,nOutputs,pPred); break;

    default: Error(std::string("Unknown Component tag:")+componentTag);
  }
 
  //read params if it is updatable component
  pRet->ReadFromStream(rIn);
  //return
  return pRet;
}


void
Network::
ComponentDumper(std::ostream& rOut, Component& rComp)
{
  //use tags of all the components; or the identification codes
  //array with list of component tags
  static const Component::ComponentType TYPES[] = {
    Component::BIASED_LINEARITY,
    Component::SHARED_LINEARITY,
    
    Component::SIGMOID,
    Component::SOFTMAX,
    Component::BLOCK_SOFTMAX,

    Component::EXPAND,
    Component::COPY,
    Component::TRANSPOSE,
    Component::BLOCK_LINEARITY,
    Component::BIAS,
    Component::WINDOW,
    Component::LOG,

    Component::BLOCK_ARRAY,
  };
  static const std::string TAGS[] = {
    "<biasedlinearity>",
    "<sharedlinearity>",

    "<sigmoid>",
    "<softmax>",
    "<blocksoftmax>",

    "<expand>",
    "<copy>",
    "<transpose>",
    "<blocklinearity>",
    "<bias>",
    "<window>",
    "<log>",

    "<blockarray>",
  };
  static const int MAX = sizeof TYPES / sizeof TYPES[0];

  int i;
  for(i=0; i<MAX; ++i) {
    if(TYPES[i] == rComp.GetType()) break;
  }
  if(i == MAX) Error("Unknown ComponentType");
  
  //dump the component tag
  rOut << TAGS[i] << " " 
       << rComp.GetNOutputs() << " " 
       << rComp.GetNInputs() << std::endl;

  //dump the parameters (if any)
  rComp.WriteToStream(rOut);
}



  
} //namespace

