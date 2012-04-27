
#ifndef _ACT_FUN_I_
#define _ACT_FUN_I_


#include "Component.h"


namespace TNet
{

  /**
   * Sigmoid activation function
   */
  class Sigmoid : public Component
  {
    public:
      Sigmoid(size_t nInputs, size_t nOutputs, Component *pPred)
       : Component(nInputs,nOutputs,pPred)
      { }

      ComponentType GetType() const
      { return SIGMOID; }

      const char* GetName() const
      { return "<sigmoid>"; }

      Component* Clone() const
      { return new Sigmoid(GetNInputs(),GetNOutputs(),NULL); }

    protected:
      void PropagateFnc(const BfMatrix& X, BfMatrix& Y);
      void BackpropagateFnc(const BfMatrix& X, BfMatrix& Y);
  };
    

  /**
   * Softmax activation function
   */
  class Softmax : public Component
  {
    public:
      Softmax(size_t nInputs, size_t nOutputs, Component *pPred)
       : Component(nInputs,nOutputs,pPred)
      { }

      ComponentType GetType() const
      { return SOFTMAX; }

      const char* GetName() const
      { return "<softmax>"; }

      Component* Clone() const
      { return new Softmax(GetNInputs(),GetNOutputs(),NULL); }

    protected:
      void PropagateFnc(const BfMatrix& X, BfMatrix& Y);
      void BackpropagateFnc(const BfMatrix& X, BfMatrix& Y);
  };


  /**
   * BlockSoftmax activation function.
   * It is several softmaxes in one.
   * The dimensions of softmaxes are given by integer vector.
   * During backpropagation: 
   *  If the derivatives sum up to 0, they are backpropagated. 
   *  If the derivatives sup up to 1, they are discarded
   *  (like this we know that the softmax was 'inactive').
   */
  class BlockSoftmax : public Component
  {
    public:
      BlockSoftmax(size_t nInputs, size_t nOutputs, Component *pPred)
       : Component(nInputs,nOutputs,pPred)
      { }

      ComponentType GetType() const
      { return BLOCK_SOFTMAX; }

      const char* GetName() const
      { return "<blocksoftmax>"; }

      Component* Clone() const
      { return new BlockSoftmax(*this); }

      void ReadFromStream(std::istream& rIn);
      void WriteToStream(std::ostream& rOut);

    protected:
      void PropagateFnc(const BfMatrix& X, BfMatrix& Y);
      void BackpropagateFnc(const BfMatrix& X, BfMatrix& Y);

    private:
      Vector<int> mDim;
      Vector<int> mDimOffset;
  };


  
} //namespace


#endif
