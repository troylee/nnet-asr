#ifndef _CUCLUSTER_LINEARITY_H_
#define _CUCLUSTER_LINEARITY_H_


#include "cuComponent.h"
#include "cumatrix.h"


#include "Matrix.h"
#include "Vector.h"

#include <vector>

namespace TNet {

  class CuClusterLinearity : public CuUpdatableComponent
  {
    public:

      CuClusterLinearity(size_t nInputs, size_t nOutputs, const char *pDir, CuComponent *pPred);
      ~CuClusterLinearity();
      
      ComponentType GetType() const;
      const char* GetName() const;

      void PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y);
      void BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y);

      void Update();

      void ReadFromStream(std::istream& rIn);
      void WriteToStream(std::ostream& rOut);

    protected:
      // The linear cluster transform
      Matrix<BaseFloat>   mClusterLinearity_host;
      Vector<BaseFloat>   mClusterBias_host;

      // The original constant weights
      Matrix<BaseFloat>   mConstLinearity_host;
      Vector<BaseFloat>   mConstBias_host;

      // The final combined weights
      Matrix<BaseFloat>   mLinearity_host;
      Vector<BaseFloat>	  mBias_host;
      CuMatrix<BaseFloat> mLinearity;  ///< Matrix with neuron weights
      CuVector<BaseFloat> mBias;       ///< Vector with biases

      CuMatrix<BaseFloat> mLinearityCorrection; ///< Matrix for linearity updates
      CuVector<BaseFloat> mBiasCorrection;      ///< Vector for bias updates

      std::vector< std::vector<int> > mClusterMap; //store the class ids of each cluster

      const char* mpTempBasisDir;
      int mNInstances;

  };




  ////////////////////////////////////////////////////////////////////////////
  // INLINE FUNCTIONS 
  // CuClusterLinearity::
  inline 
  CuClusterLinearity::
  CuClusterLinearity(size_t nInputs, size_t nOutputs, const char *pDir, CuComponent *pPred)
    : CuUpdatableComponent(nInputs, nOutputs, pPred), 
      mClusterLinearity_host(nInputs, nInputs), mClusterBias_host(nInputs),
      mConstLinearity_host(nInputs, nOutputs), mConstBias_host(nOutputs),
      mLinearity_host(nInputs, nOutputs), mBias_host(nOutputs),
      mLinearity(nInputs,nOutputs), mBias(nOutputs),
      mLinearityCorrection(nInputs,nOutputs), mBiasCorrection(nOutputs), mpTempBasisDir(pDir), mNInstances(0)
  { 
    mLinearityCorrection.SetConst(0.0);
    mBiasCorrection.SetConst(0.0);
    mClusterMap.clear();
  }


  inline
  CuClusterLinearity::
  ~CuClusterLinearity()
  { }

  inline CuComponent::ComponentType
  CuClusterLinearity::
  GetType() const
  {
    return CuComponent::CLUSTER_LINEARITY;
  }

  inline const char*
  CuClusterLinearity::
  GetName() const
  {
    return "<clusterlinearity>";
  }



} //namespace



#endif
