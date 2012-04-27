#ifndef _CUNETWORK_H_
#define _CUNETWORK_H_

#include "cuComponent.h"

#include "cuBiasedLinearity.h"
//#include "cuBlockLinearity.h"
//#include "cuBias.h"
//#include "cuWindow.h"

#include "cuActivation.h"

#include "cuCRBEDctFeat.h"

#include "Vector.h"

#include <vector>


namespace TNet {

  class CuNetwork
  {
    //////////////////////////////////////
    // Typedefs
    typedef std::vector<CuComponent*> LayeredType;
      
      //////////////////////////////////////
      // Disable copy construction, assignment and default constructor
    private:
      CuNetwork(CuNetwork&); 
      CuNetwork& operator=(CuNetwork&);
       
    public:
      CuNetwork() { }
      CuNetwork(std::istream& rIn); 
      ~CuNetwork();

      void AddLayer(CuComponent* layer);

      int Layers()
      { return mNetComponents.size(); }

      CuComponent& Layer(int i)
      { return *mNetComponents[i]; }

      /// forward the data to the output
      void Propagate(const CuMatrix<BaseFloat>& in, CuMatrix<BaseFloat>& out);

      /// backpropagate the error while updating weights
      void Backpropagate(const CuMatrix<BaseFloat>& globerr); 

      void ReadNetwork(const char* pSrc);     ///< read the network from file
      void WriteNetwork(const char* pDst);    ///< write network to file

      void ReadNetwork(std::istream& rIn);    ///< read the network from stream
      void WriteNetwork(std::ostream& rOut);  ///< write network to stream

      size_t GetNInputs() const; ///< Dimensionality of the input features
      size_t GetNOutputs() const; ///< Dimensionality of the desired vectors

      /// set the learning rate
      void SetLearnRate(BaseFloat learnRate, const char* pLearnRateFactors = NULL); 
      BaseFloat GetLearnRate();  ///< get the learning rate value
      void PrintLearnRate();     ///< log the learning rate values

      void SetMomentum(BaseFloat momentum);
      void SetWeightcost(BaseFloat weightcost);
      void SetL1(BaseFloat l1);

      void SetGradDivFrm(bool div);

      void SetTempBasisDir(const char* pDir); 	///< set the dir for storing temp basis/xforms

    private:
      /// Creates a component by reading from stream
      CuComponent* ComponentFactory(std::istream& In);
      /// Dumps component into a stream
      void ComponentDumper(std::ostream& rOut, CuComponent& rComp);



    private:
      LayeredType mNetComponents; ///< container with the network layers
      CuComponent* mpPropagErrorStopper;
      BaseFloat mGlobLearnRate; ///< The global (unscaled) learn rate of the network
      const char* mpLearnRateFactors; ///< The global (unscaled) learn rate of the network
      const char* mpTempBasisDir; ///< the folder to save temp basis/xform
      

    //friend class NetworkGenerator; //<< For generating networks...

  };
    

  //////////////////////////////////////////////////////////////////////////
  // INLINE FUNCTIONS 
  // CuNetwork::
  inline 
  CuNetwork::
  CuNetwork(std::istream& rSource)
    : mpPropagErrorStopper(NULL), mGlobLearnRate(0.0), mpLearnRateFactors(NULL), mpTempBasisDir(NULL)
  {
    ReadNetwork(rSource);
  }


  inline
  CuNetwork::
  ~CuNetwork()
  {
    //delete all the components
    LayeredType::iterator it;
    for(it=mNetComponents.begin(); it!=mNetComponents.end(); ++it) {
      delete *it;
      *it = NULL;
    }
    mNetComponents.resize(0);
  }

  
  inline void 
  CuNetwork::
  AddLayer(CuComponent* layer)
  {
    if(mNetComponents.size() > 0) {
      if(GetNOutputs() != layer->GetNInputs()) {
        Error("Nonmatching dims");
      }
      layer->SetInput(mNetComponents.back()->GetOutput());
      mNetComponents.back()->SetErrorInput(layer->GetErrorOutput());
    }
    mNetComponents.push_back(layer);
  }


  inline void
  CuNetwork::
  Propagate(const CuMatrix<BaseFloat>& in, CuMatrix<BaseFloat>& out)
  {
    //empty network => copy input
    if(mNetComponents.size() == 0) { 
      out.CopyFrom(in); 
      return;
    }

    //check dims
    if(in.Cols() != GetNInputs()) {
      std::ostringstream os;
      os << "Nonmatching dims"
         << " data dim is: " << in.Cols() 
         << " network needs: " << GetNInputs();
      Error(os.str());
    }
    mNetComponents.front()->SetInput(in);
    
    //propagate
    LayeredType::iterator it;
    for(it=mNetComponents.begin(); it!=mNetComponents.end(); ++it) {
      (*it)->Propagate();
    }

    //copy the output
    out.CopyFrom(mNetComponents.back()->GetOutput());
  }




  inline void 
  CuNetwork::
  Backpropagate(const CuMatrix<BaseFloat>& globerr) 
  {
    mNetComponents.back()->SetErrorInput(globerr);

    // back-propagation
    LayeredType::reverse_iterator it;
    for(it=mNetComponents.rbegin(); it!=mNetComponents.rend(); ++it) {
      //stopper component does not propagate error (no updatable predecessors)
      if(*it != mpPropagErrorStopper) {
        //compute errors for preceding network components
        (*it)->Backpropagate();
      }
      //update weights if updatable component
      if((*it)->IsUpdatable()) {
        CuUpdatableComponent& rComp = dynamic_cast<CuUpdatableComponent&>(**it); 
        if(rComp.LearnRate() > 0.0f) {
          rComp.Update();
        }
      }
      //stop backprop if no updatable components precede current component
      if(mpPropagErrorStopper == *it) break;
    }
  }

      
  inline size_t
  CuNetwork::
  GetNInputs() const
  {
    if(!mNetComponents.size() > 0) return 0;
    return mNetComponents.front()->GetNInputs();
  }


  inline size_t
  CuNetwork::
  GetNOutputs() const
  {
    if(!mNetComponents.size() > 0) return 0;
    return mNetComponents.back()->GetNOutputs();
  }





} //namespace

#endif


