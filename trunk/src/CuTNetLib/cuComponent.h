#ifndef _CUNETWORK_COMPONENT_I_H
#define _CUNETWORK_COMPONENT_I_H


#include "Vector.h"
#include "Matrix.h"
#include "Error.h"

#include "cumatrix.h"

#include <iostream>
#include <stdexcept>


namespace TNet {

    
  /**
   * Basic element of the network,
   * it is a box with defined inputs and outputs, 
   * and functions to refresh outputs
   *
   * it is able to compute tranformation function (forward pass) 
   * and jacobian function (backward pass), 
   * which is to be implemented in descendents
   */ 
  class CuComponent 
  {
    public:
    /// Types of the net components
    typedef enum { 
      UPDATABLE_COMPONENT = 0x0100, 
      BIASED_LINEARITY,
      DISCRETE_LINEARITY,
      SHARED_LINEARITY,
      SPARSE_LINEARITY,
      RBM,
      RBM_SPARSE,
      RECURRENT,

      ACT_FUN = 0x0200, 
      SOFTMAX, 
      SIGMOID,

      OTHER = 0x0400,
      EXPAND,
      COPY,
      TRANSPOSE,
      BLOCK_LINEARITY,
      WINDOW,
      BIAS,
      LOG,

      BLOCK_ARRAY,

      CLUSTER_LINEARITY,

    } ComponentType;

    //////////////////////////////////////////////////////////////
    // Constructor & Destructor
    public: 
      CuComponent(size_t nInputs, size_t nOutputs, CuComponent *pPred); 
      virtual ~CuComponent();  
       
    //////////////////////////////////////////////////////////////
    // Interface specification (public)
    public:
      /// Get Type Identification of the component
      virtual ComponentType GetType() const = 0;  
      /// Get Type Label of the component
      virtual const char* GetName() const = 0;
      /// 
      virtual bool IsUpdatable() const 
      { return false; }

      /// Get size of input vectors
      size_t GetNInputs() const;  
      /// Get size of output vectors 
      size_t GetNOutputs() const; 
     
      /// IO Data getters
      const CuMatrix<BaseFloat>& GetInput() const; 
      const CuMatrix<BaseFloat>& GetOutput() const;
      const CuMatrix<BaseFloat>& GetErrorInput() const;
      const CuMatrix<BaseFloat>& GetErrorOutput() const;
      
      /// Set input vector (bind with the preceding NetworkComponent)
      void SetInput(const CuMatrix<BaseFloat>& rInput);           
      /// Set error input vector (bind with the following NetworkComponent) 
      void SetErrorInput(const CuMatrix<BaseFloat>& rErrorInput);  
       
      /// Perform forward pass propagateion Input->Output
      void Propagate(); 
      /// Perform backward pass propagateion ErrorInput->ErrorOutput
      void Backpropagate(); 
 
      /// Reads the component parameters from stream
      virtual void ReadFromStream(std::istream& rIn)  { }
      /// Writes the components parameters to stream
      virtual void WriteToStream(std::ostream& rOut)  { } 


    ///////////////////////////////////////////////////////////////
    // Nonpublic member functions used to update data outputs 
    protected:
      /// Forward pass transformation (to be implemented by descendents...)
      virtual void PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y) = 0;
      /// Backward pass transformation (to be implemented by descendents...)
      virtual void BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y) = 0;
      
   
    ///////////////////////////////////////////////////////////////
    // data members
    protected:

      size_t mNInputs;  ///< Size of input vectors
      size_t mNOutputs; ///< Size of output vectors 
      
      const CuMatrix<BaseFloat>* mpInput; ///< inputs are NOT OWNED by component
      const CuMatrix<BaseFloat>* mpErrorInput;///< inputs are NOT OWNED by component

      CuMatrix<BaseFloat> mOutput; ///< outputs are OWNED by component
      CuMatrix<BaseFloat> mErrorOutput; ///< outputs are OWNED by component

  };


  /**
   * Class UpdatableComponent is a box which has some 
   * parameters adjustable by learning
   *
   * you can set the learning rate, lock the params,
   * and learn from each data observation
   */
  class CuUpdatableComponent : public CuComponent
  {
    //////////////////////////////////////////////////////////////
    // Constructor & Destructor
    public: 
      CuUpdatableComponent(size_t nInputs, size_t nOutputs, CuComponent *pPred); 
      virtual ~CuUpdatableComponent();


    //////////////////////////////////////////////////////////////
    // Interface specification (public)
    public:
      ///
      virtual bool IsUpdatable() const 
      { return true; }

      /// get gradient and update the parameters in one step
      virtual void Update() = 0;    
      
      /// Sets the learning rate of gradient descent
      void LearnRate(BaseFloat rate);
      /// Gets the learning rate of gradient descent
      BaseFloat LearnRate();

      void Momentum(BaseFloat mmt);
      BaseFloat Momentum();

      void Weightcost(BaseFloat cost);
      BaseFloat Weightcost();

      void GradDivFrm(bool div);
      bool GradDivFrm();
  
    protected:
      BaseFloat mLearningRate;
      BaseFloat mMomentum;
      BaseFloat mWeightcost;
      bool      mGradDivFrm;

  };




  //////////////////////////////////////////////////////////////////////////
  // INLINE FUNCTIONS 
  // CuComponent::
  inline
  CuComponent::
  CuComponent(size_t nInputs, size_t nOutputs, CuComponent *pPred) 
    : mNInputs(nInputs), mNOutputs(nOutputs), 
      mpInput(NULL), mpErrorInput(NULL), 
      mOutput(), mErrorOutput()
  { 
    /* DOUBLE LINK the Components */
    if (pPred != NULL) {
      SetInput(pPred->GetOutput());
      pPred->SetErrorInput(GetErrorOutput());
    }
  } 


  inline
  CuComponent::
  ~CuComponent()
  {
    ;
  }

  inline void
  CuComponent::
  Propagate()
  {
    //initialize output buffer
    mOutput.Init(GetInput().Rows(),GetNOutputs());
    //do the dimensionality test
    if(GetNInputs() != GetInput().Cols()) {
      KALDI_ERR << "Non-matching INPUT dim!!! Network dim: " << GetNInputs() 
                << " Data dim: " << GetInput().Cols();
    }
    //run transform
    PropagateFnc(GetInput(),mOutput);
  }


  inline void
  CuComponent::
  Backpropagate()
  {
    //re-initialize the output buffer
    mErrorOutput.Init(GetErrorInput().Rows(),GetNInputs());

    //do the dimensionality test
    assert(GetErrorInput().Cols() == mNOutputs);
    assert(mErrorOutput.Cols() == mNInputs);
    assert(mErrorOutput.Rows() == GetErrorInput().Rows());

    //transform
    BackpropagateFnc(GetErrorInput(),mErrorOutput);
  }


  inline void
  CuComponent::
  SetInput(const CuMatrix<BaseFloat>& rInput)
  {
    mpInput = &rInput;
  }


  inline void
  CuComponent::
  SetErrorInput(const CuMatrix<BaseFloat>& rErrorInput)
  {
    mpErrorInput = &rErrorInput;
  }


  inline const CuMatrix<BaseFloat>&
  CuComponent::
  GetInput() const
  {
    if (NULL == mpInput) Error("mpInput is NULL");
    return *mpInput;
  }

  inline const CuMatrix<BaseFloat>&
  CuComponent::
  GetOutput() const
  {
    return mOutput;
  }

  inline const CuMatrix<BaseFloat>&
  CuComponent::
  GetErrorInput() const
  {
    if (NULL == mpErrorInput) Error("mpErrorInput is NULL");
    return *mpErrorInput;
  }

  inline const CuMatrix<BaseFloat>&
  CuComponent::
  GetErrorOutput() const
  {
    return mErrorOutput;
  }

  inline size_t
  CuComponent::
  GetNInputs() const
  {
    return mNInputs;
  }

  inline size_t
  CuComponent::
  GetNOutputs() const
  {
    return mNOutputs;
  }



  //////////////////////////////////////////////////////////////////////////
  // INLINE FUNCTIONS 
  // UpdatableComponent::
  
  inline 
  CuUpdatableComponent::
  CuUpdatableComponent(size_t nInputs, size_t nOutputs, CuComponent *pPred) 
    : CuComponent(nInputs, nOutputs, pPred), 
      mLearningRate(0.0), mMomentum(0), mWeightcost(0), mGradDivFrm(true)
  {
    ; 
  } 


  inline
  CuUpdatableComponent::
  ~CuUpdatableComponent()
  {
    ;
  }


  inline void
  CuUpdatableComponent::
  LearnRate(BaseFloat rate)
  {
    mLearningRate = rate;
  }


  inline BaseFloat
  CuUpdatableComponent::
  LearnRate()
  {
    return mLearningRate;
  }
  

  inline void
  CuUpdatableComponent::
  Momentum(BaseFloat mmt)
  {
    mMomentum = mmt;
  }


  inline BaseFloat
  CuUpdatableComponent::
  Momentum()
  {
    return mMomentum;
  }
  
  
  inline void
  CuUpdatableComponent::
  Weightcost(BaseFloat cost)
  {
    mWeightcost = cost;
  }


  inline BaseFloat
  CuUpdatableComponent::
  Weightcost()
  {
    return mWeightcost;
  }

  
  inline void 
  CuUpdatableComponent::
  GradDivFrm(bool div)
  {
    mGradDivFrm = div;
  }
   
  inline bool
  CuUpdatableComponent::
  GradDivFrm()
  {
    return mGradDivFrm;
  }

} // namespace TNet


#endif
