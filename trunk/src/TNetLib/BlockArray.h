#ifndef _BLOCK_ARRAY_H_
#define _BLOCK_ARRAY_H_


#include "Component.h"

#include "Matrix.h"
#include "Vector.h"


namespace TNet {

  class Network;

  class BlockArray : public Component
  {
    public:

      BlockArray(size_t nInputs, size_t nOutputs, Component *pPred); 
      ~BlockArray();  
      
      ComponentType GetType() const;
      const char* GetName() const;

      void PropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y);
      void BackpropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y);

      void Update();

      void ReadFromStream(std::istream& rIn);
      void WriteToStream(std::ostream& rOut);
 
      //:TODO:
      Component* Clone() const { KALDI_ERR << "Unimplemented"; }

    protected:
      std::vector<Network*> mBlocks; ///< vector with networks, one network is one block
      size_t mNBlocks;  
  };




  ////////////////////////////////////////////////////////////////////////////
  // INLINE FUNCTIONS 
  // BlockArray::
  inline 
  BlockArray::
  BlockArray(size_t nInputs, size_t nOutputs, Component *pPred)
    : Component(nInputs, nOutputs, pPred), 
      mNBlocks(0) 
  { }


  inline
  BlockArray::
  ~BlockArray()
  { 
    for(int i=0; i<mBlocks.size(); i++) {
      delete mBlocks[i];
    }
    mBlocks.clear();
  }

  inline Component::ComponentType
  BlockArray::
  GetType() const
  {
    return Component::BLOCK_ARRAY;
  }

  inline const char*
  BlockArray::
  GetName() const
  {
    return "<blockarray>";
  }



} //namespace



#endif
