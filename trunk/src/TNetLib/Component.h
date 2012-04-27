#ifndef _NETWORK_COMPONENT_I_H
#define _NETWORK_COMPONENT_I_H


#include "Vector.h"
#include "Matrix.h"

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
  class Component 
  {
    public:
    /// Types of the net components
    typedef enum { 
      UPDATABLE_COMPONENT = 0x0100, 
      BIASED_LINEARITY,
      SHARED_LINEARITY,

      ACT_FUN = 0x0200, 
      SOFTMAX,
      SIGMOID,
      BLOCK_SOFTMAX, 

      OTHER = 0x0400,
      EXPAND,
      COPY,
      TRANSPOSE,
      BLOCK_LINEARITY,
      WINDOW,
      BIAS,
      LOG,
      
      BLOCK_ARRAY,
    } ComponentType;


    //////////////////////////////////////////////////////////////
    // Constructor & Destructor
    public: 
      Component(size_t nInputs, size_t nOutputs, Component *pPred); 
      virtual ~Component();  
       
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
      /// Clone the component
      virtual Component* Clone() const = 0; 

      /// Get size of input vectors
      size_t GetNInputs() const;  
      /// Get size of output vectors 
      size_t GetNOutputs() const; 
     
      /// IO Data getters
      const Matrix<BaseFloat>& GetInput() const; 
      const Matrix<BaseFloat>& GetOutput() const;
      const Matrix<BaseFloat>& GetErrorInput() const;
      const Matrix<BaseFloat>& GetErrorOutput() const;
      
      /// Set input vector (bind with the preceding NetworkComponent)
      void SetInput(const Matrix<BaseFloat>& rInput);           
      /// Set error input vector (bind with the following NetworkComponent) 
      void SetErrorInput(const Matrix<BaseFloat>& rErrorInput);  
       
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
      virtual void PropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y) = 0;
      /// Backward pass transformation (to be implemented by descendents...)
      virtual void BackpropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y) = 0;

   
    ///////////////////////////////////////////////////////////////
    // data members
    protected:

      size_t mNInputs;  ///< Size of input vectors
      size_t mNOutputs; ///< Size of output vectors 
      
      const Matrix<BaseFloat>* mpInput; ///< inputs are NOT OWNED by component
      const Matrix<BaseFloat>* mpErrorInput;///< inputs are NOT OWNED by component

      Matrix<BaseFloat> mOutput; ///< outputs are OWNED by component
      Matrix<BaseFloat> mErrorOutput; ///< outputs are OWNED by component

  };


  /**
   * Class UpdatableComponent is a box which has some 
   * parameters adjustable by learning
   *
   * you can set the learning rate, lock the params,
   * and learn from each data observation
   */
  class UpdatableComponent : public Component
  {
    //////////////////////////////////////////////////////////////
    // Constructor & Destructor
    public: 
      UpdatableComponent(size_t nInputs, size_t nOutputs, Component *pPred); 
      virtual ~UpdatableComponent();


    //////////////////////////////////////////////////////////////
    // Interface specification (public)
    public:
      ///
      virtual bool IsUpdatable() const 
      { return true; }

      /// calculate gradient
      virtual void Gradient() = 0;
      /// accumulate gradient from other components
      virtual void AccuGradient(const UpdatableComponent& src, int thr, int thrN) = 0;  
      /// update weights, reset the accumulator
      virtual void Update(int thr, int thrN) = 0;

      /// Sets the learning rate of gradient descent
      void LearnRate(BaseFloat rate);
      /// Gets the learning rate of gradient descent
      BaseFloat LearnRate() const;

      void Momentum(BaseFloat mmt);
      BaseFloat Momentum() const ;

      void Weightcost(BaseFloat cost);
      BaseFloat Weightcost() const;

      void Bunchsize(size_t size);
      size_t Bunchsize() const;

    protected:
      BaseFloat mLearningRate;
      BaseFloat mMomentum;
      BaseFloat mWeightcost;
      size_t mBunchsize;
  };




  //////////////////////////////////////////////////////////////////////////
  // INLINE FUNCTIONS 
  // Component::
  inline
  Component::
  Component(size_t nInputs, size_t nOutputs, Component *pPred) 
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
  Component::
  ~Component()
  {
    ;
  }

  inline void
  Component::
  Propagate()
  {
    //initialize output buffer
    if(mOutput.Rows() != GetInput().Rows() || mOutput.Cols() != GetNOutputs()) {
      mOutput.Init(GetInput().Rows(),GetNOutputs());
    }
    //do the dimensionality test
    if(GetNInputs() != GetInput().Cols()) {
      KALDI_ERR << "Non-matching INPUT dim!!! Network dim: " << GetNInputs() 
                << " Data dim: " << GetInput().Cols();
    }
    //run transform
    PropagateFnc(GetInput(),mOutput);
  
  }


  inline void
  Component::
  Backpropagate()
  {
    //re-initialize the output buffer
    if(mErrorOutput.Rows() != GetErrorInput().Rows() || mErrorOutput.Cols() != GetNInputs()) {
      mErrorOutput.Init(GetErrorInput().Rows(),GetNInputs());
    }

    //do the dimensionality test
    assert(GetErrorInput().Cols() == mNOutputs);
    assert(mErrorOutput.Cols() == mNInputs);
    assert(mErrorOutput.Rows() == GetErrorInput().Rows());

    //transform
    BackpropagateFnc(GetErrorInput(),mErrorOutput);

 }


  inline void
  Component::
  SetInput(const Matrix<BaseFloat>& rInput)
  {
    mpInput = &rInput;
  }


  inline void
  Component::
  SetErrorInput(const Matrix<BaseFloat>& rErrorInput)
  {
    mpErrorInput = &rErrorInput;
  }


  inline const Matrix<BaseFloat>&
  Component::
  GetInput() const
  {
    if (NULL == mpInput) Error("mpInput is NULL");
    return *mpInput;
  }

  inline const Matrix<BaseFloat>&
  Component::
  GetOutput() const
  {
    return mOutput;
  }

  inline const Matrix<BaseFloat>&
  Component::
  GetErrorInput() const
  {
    if (NULL == mpErrorInput) Error("mpErrorInput is NULL");
    return *mpErrorInput;
  }

  inline const Matrix<BaseFloat>&
  Component::
  GetErrorOutput() const
  {
    return mErrorOutput;
  }

  inline size_t
  Component::
  GetNInputs() const
  {
    return mNInputs;
  }

  inline size_t
  Component::
  GetNOutputs() const
  {
    return mNOutputs;
  }



  //////////////////////////////////////////////////////////////////////////
  // INLINE FUNCTIONS 
  // UpdatableComponent::
  
  inline 
  UpdatableComponent::
  UpdatableComponent(size_t nInputs, size_t nOutputs, Component *pPred) 
    : Component(nInputs, nOutputs, pPred), 
      mLearningRate(0.0), mMomentum(0.0), mWeightcost(0.0), mBunchsize(0)
  {
    ; 
  } 


  inline
  UpdatableComponent::
  ~UpdatableComponent()
  {
    ;
  }


  inline void
  UpdatableComponent::
  LearnRate(BaseFloat rate)
  {
    mLearningRate = rate;
  }

  inline BaseFloat
  UpdatableComponent::
  LearnRate() const
  {
    return mLearningRate;
  }


  inline void
  UpdatableComponent::
  Momentum(BaseFloat mmt)
  {
    mMomentum = mmt;
  }

  inline BaseFloat
  UpdatableComponent::
  Momentum() const
  {
    return mMomentum;
  }
  
  
  inline void
  UpdatableComponent::
  Weightcost(BaseFloat cost)
  {
    mWeightcost = cost;
  }

  inline BaseFloat
  UpdatableComponent::
  Weightcost() const
  {
    return mWeightcost;
  }

  
  inline void
  UpdatableComponent::
  Bunchsize(size_t size)
  {
    mBunchsize = size;
  }

  inline size_t
  UpdatableComponent::
  Bunchsize() const
  {
    return mBunchsize;
  }


} // namespace TNet


#endif
