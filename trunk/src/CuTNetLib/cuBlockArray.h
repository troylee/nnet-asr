#ifndef _CUBLOCK_ARRAY_H_
#define _CUBLOCK_ARRAY_H_


#include "cuComponent.h"
#include "cumatrix.h"

#include "Matrix.h"
#include "Vector.h"


namespace TNet {

  class CuNetwork;

  class CuBlockArray : public CuUpdatableComponent
  {
    public:

      CuBlockArray(size_t nInputs, size_t nOutputs, CuComponent *pPred); 
      ~CuBlockArray();  
      
      ComponentType GetType() const;
      const char* GetName() const;

      void PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y);
      void BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y);

      void Update();

      void ReadFromStream(std::istream& rIn);
      void WriteToStream(std::ostream& rOut);

    protected:
      std::vector<CuNetwork*> mBlocks; ///< vector with networks, one network is one block
      size_t mNBlocks;  
  };




  ////////////////////////////////////////////////////////////////////////////
  // INLINE FUNCTIONS 
  // CuBlockArray::
  inline 
  CuBlockArray::
  CuBlockArray(size_t nInputs, size_t nOutputs, CuComponent *pPred)
    : CuUpdatableComponent(nInputs, nOutputs, pPred), 
      mNBlocks(0) 
  { }


  inline
  CuBlockArray::
  ~CuBlockArray()
  { 
    for(int i=0; i<mBlocks.size(); i++) {
      delete mBlocks[i];
    }
    mBlocks.clear();
  }

  inline CuComponent::ComponentType
  CuBlockArray::
  GetType() const
  {
    return CuComponent::BLOCK_ARRAY;
  }

  inline const char*
  CuBlockArray::
  GetName() const
  {
    return "<blockarray>";
  }



} //namespace



#endif
