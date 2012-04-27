
#include <algorithm>
//#include <locale>
#include <cctype>
#include <list>
#include <sstream>

#include "cuNetwork.h"

#include "cuDiscreteLinearity.h"
#include "cuSharedLinearity.h"
#include "cuSparseLinearity.h"
#include "cuRbm.h"
#include "cuRbmSparse.h"
#include "cuRecurrent.h"
#include "cuBlockArray.h"
#include "cuClusterLinearity.h"

namespace TNet {

	void CuNetwork::SetTempBasisDir(const char* pDir)
	{
		mpTempBasisDir=pDir;
	}
  

  void
  CuNetwork::
  ReadNetwork(const char* pSrc)
  {
    std::ifstream in(pSrc);
    if(!in.good()) {
      Error(std::string("Error, cannot read model: ")+pSrc);
    }
    ReadNetwork(in);
    in.close();
  }

 
 
  void
  CuNetwork::
  WriteNetwork(const char* pDst)
  {
    std::ofstream out(pDst);
    if(!out.good()) {
      Error(std::string("Error, cannot write model: ")+pDst);
    }
    WriteNetwork(out);
    out.close();
  }

   

  void
  CuNetwork::
  ReadNetwork(std::istream& rIn)
  {
    //get the network elements from a factory
    CuComponent *pComp;
    while(NULL != (pComp = ComponentFactory(rIn))) { 
      mNetComponents.push_back(pComp);
    }
  }



  void
  CuNetwork::
  WriteNetwork(std::ostream& rOut)
  {
    //dump all the componetns
    LayeredType::iterator it;
    for(it=mNetComponents.begin(); it!=mNetComponents.end(); ++it) {
      ComponentDumper(rOut, **it);
    }
  }


  void
  CuNetwork::
  SetLearnRate(BaseFloat learnRate, const char* pLearnRateFactors)
  {
    //parse the learn rate factors: "0.1:0.5:0.6:1.0" to std::list
    std::list<BaseFloat> lr_factors;
    if(NULL != pLearnRateFactors) {
      //replace ':' by ' '
      std::string str(pLearnRateFactors);
      size_t pos = 0;
      while((pos = str.find(':',pos)) != std::string::npos) str[pos] = ' ';
      while((pos = str.find(',',pos)) != std::string::npos) str[pos] = ' ';

      //parse to std::list
      std::istringstream is(str);
      is >> std::skipws;
      BaseFloat f; 
      while(!is.eof()) {
        if(!(is >> f).fail()) { lr_factors.push_back(f); }
        else break;
      }
    }

    //initialize rate factors iterator
    BaseFloat scale = 1.0f;

    //store global learning rate
    mGlobLearnRate = learnRate;
    mpLearnRateFactors = pLearnRateFactors;

    //give scaled learning rate to components
    LayeredType::iterator it;
    bool stopper_given = false;
    for(it=mNetComponents.begin(); it!=mNetComponents.end(); ++it) {
      if((*it)->IsUpdatable()) {
        //get next scale factor
        if(NULL != pLearnRateFactors) {
          if(!(lr_factors.size() > 0)) {
            Error("Too few learninig rate scale factors");
          }
          scale = lr_factors.front(); 
          lr_factors.pop_front(); 
        }
        //set scaled learning rate to the component
        dynamic_cast<CuUpdatableComponent*>(*it)->LearnRate(learnRate*scale);
        //set the stopper component for backpropagation
        if(!stopper_given && (learnRate*scale > 0.0)) {
          mpPropagErrorStopper = *it; stopper_given = true;
        }
      }
    }
    if(lr_factors.size() > 0) {
      Error("Too much learninig rate scale factors");
    }
  }


  BaseFloat
  CuNetwork::
  GetLearnRate()
  {
    return mGlobLearnRate;
  }


  void
  CuNetwork::
  PrintLearnRate()
  {
    assert(mNetComponents.size() > 0);
    std::cout << "Learning rate: global " << mGlobLearnRate;
    std::cout << " components' ";
    for(size_t i=0; i<mNetComponents.size(); i++) {
      if(mNetComponents[i]->IsUpdatable()) {
        std::cout << " " << dynamic_cast<CuUpdatableComponent*>(mNetComponents[i])->LearnRate();
      }
    }
    std::cout << "\n" << std::flush;
  }



  void
  CuNetwork::
  SetMomentum(BaseFloat momentum)
  {
    LayeredType::iterator it;
    for(it=mNetComponents.begin(); it!=mNetComponents.end(); ++it) {
      if((*it)->IsUpdatable()) {
        dynamic_cast<CuUpdatableComponent*>(*it)->Momentum(momentum);
      }
    }
  }

  void
  CuNetwork::
  SetWeightcost(BaseFloat weightcost)
  {
    LayeredType::iterator it;
    for(it=mNetComponents.begin(); it!=mNetComponents.end(); ++it) {
      if((*it)->IsUpdatable()) {
        dynamic_cast<CuUpdatableComponent*>(*it)->Weightcost(weightcost);
      }
    }
  }

  void
  CuNetwork::
  SetL1(BaseFloat l1)
  {
    LayeredType::iterator it;
    for(it=mNetComponents.begin(); it!=mNetComponents.end(); ++it) {
      if((*it)->GetType() == CuComponent::SPARSE_LINEARITY) {
        dynamic_cast<CuSparseLinearity*>(*it)->L1(l1);
      }
    }
  }

  void
  CuNetwork::
  SetGradDivFrm(bool div)
  {
    LayeredType::iterator it;
    for(it=mNetComponents.begin(); it!=mNetComponents.end(); ++it) {
      if((*it)->IsUpdatable()) {
        dynamic_cast<CuUpdatableComponent*>(*it)->GradDivFrm(div);
      }
    }
  }
   

  CuComponent*
  CuNetwork::
  ComponentFactory(std::istream& rIn)
  {
    rIn >> std::ws;
    if(rIn.eof()) return NULL;

    CuComponent* pRet=NULL;
    CuComponent* pPred=NULL;

    std::string componentTag;
    size_t nInputs, nOutputs;

    rIn >> std::ws;
    rIn >> componentTag;
    if(componentTag == "") return NULL; //nothing left in the file

    //make it lowercase
    std::transform(componentTag.begin(), componentTag.end(), 
                   componentTag.begin(), tolower);

    if(componentTag[0] != '<' || componentTag[componentTag.size()-1] != '>') {
      Error(std::string("Invalid component tag:")+componentTag);
    }

    //the 'endblock' tag terminates the network
    if(componentTag == "<endblock>") return NULL;

    rIn >> std::ws;
    rIn >> nOutputs;
    rIn >> std::ws;
    rIn >> nInputs;
    assert(nInputs > 0 && nOutputs > 0);

    //make coupling with predecessor
    if(mNetComponents.size() != 0) {
      pPred = mNetComponents.back();
    }
    
    //array with list of component tags
    static const std::string TAGS[] = {
      "<biasedlinearity>",
      "<discretelinearity>",
      "<sharedlinearity>",
      "<sparselinearity>",
      "<rbm>",
      "<rbmsparse>",
      "<recurrent>",

      "<softmax>",
      "<sigmoid>",

      "<expand>",
      "<copy>",
      "<transpose>",
      "<blocklinearity>",
      "<bias>",
      "<window>",
      "<log>", 

      "<blockarray>",

      "<clusterlinearity>",
    };

    static const int n_tags = sizeof(TAGS) / sizeof(TAGS[0]);
    int i;
    for(i=0; i<n_tags; i++) {
      if(componentTag == TAGS[i]) break;
    }
       
    //switch according to position in array TAGS
    switch(i) {
      case 0: pRet = new CuBiasedLinearity(nInputs,nOutputs,pPred); break;
      case 1: pRet = new CuDiscreteLinearity(nInputs,nOutputs,pPred); break;
      case 2: pRet = new CuSharedLinearity(nInputs,nOutputs,pPred); break;
      case 3: pRet = new CuSparseLinearity(nInputs,nOutputs,pPred); break;
      case 4: pRet = new CuRbm(nInputs,nOutputs,pPred); break;
      case 5: pRet = new CuRbmSparse(nInputs,nOutputs,pPred); break;
      case 6: pRet = new CuRecurrent(nInputs,nOutputs,pPred); break;

      case 7: pRet = new CuSoftmax(nInputs,nOutputs,pPred); break;
      case 8: pRet = new CuSigmoid(nInputs,nOutputs,pPred); break;

      case 9: pRet = new CuExpand(nInputs,nOutputs,pPred); break;
      case 10: pRet = new CuCopy(nInputs,nOutputs,pPred); break;
      case 11: pRet = new CuTranspose(nInputs,nOutputs,pPred); break;
      case 12: pRet = new CuBlockLinearity(nInputs,nOutputs,pPred); break;
      case 13: pRet = new CuBias(nInputs,nOutputs,pPred); break;
      case 14: pRet = new CuWindow(nInputs,nOutputs,pPred); break;
      case 15: pRet = new CuLog(nInputs,nOutputs,pPred); break;
     
      case 16: pRet = new CuBlockArray(nInputs,nOutputs,pPred); break;
      
      case 17: pRet = new CuClusterLinearity(nInputs, nOutputs, mpTempBasisDir, pPred); break;

      default: Error(std::string("Unknown Component tag:")+componentTag);
    }
   
    //read components content
    pRet->ReadFromStream(rIn);
        
    //return
    return pRet;
  }


  void
  CuNetwork::
  ComponentDumper(std::ostream& rOut, CuComponent& rComp)
  {
    //use tags of all the components; or the identification codes
    //array with list of component tags
    static const CuComponent::ComponentType TYPES[] = {
      CuComponent::BIASED_LINEARITY,
      CuComponent::DISCRETE_LINEARITY,
      CuComponent::SHARED_LINEARITY,
      CuComponent::SPARSE_LINEARITY,
      CuComponent::RBM,
      CuComponent::RBM_SPARSE,
      CuComponent::RECURRENT,

      CuComponent::SIGMOID,
      CuComponent::SOFTMAX,

      CuComponent::EXPAND,
      CuComponent::COPY,
      CuComponent::TRANSPOSE,
      CuComponent::BLOCK_LINEARITY,
      CuComponent::BIAS,
      CuComponent::WINDOW,
      CuComponent::LOG,

      CuComponent::BLOCK_ARRAY,

      CuComponent::CLUSTER_LINEARITY,
    };
    static const std::string TAGS[] = {
      "<biasedlinearity>",
      "<discretelinearity>",
      "<sharedlinearity>",
      "<sparselinearity>",
      "<rbm>",
      "<rbmsparse>",
      "<recurrent>",

      "<sigmoid>",
      "<softmax>",

      "<expand>",
      "<copy>",
      "<transpose>",
      "<blocklinearity>",
      "<bias>",
      "<window>",
      "<log>",

      "<blockarray>",

      "<clusterlinearity>",
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

    //write components content
    rComp.WriteToStream(rOut);
  }


  
} //namespace

