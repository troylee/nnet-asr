#ifndef _CUCACHE_H_
#define _CUCACHE_H_

#include "Matrix.h"

namespace TNet {


  /**
   * The feature-target pair cache
   */
  class Cache {
    typedef enum { EMPTY, INTAKE, FULL, EXHAUST } State;
    public:
      Cache();
      ~Cache();
     
      /// Initialize the cache
      void Init(size_t cachesize, size_t bunchsize, long int seed = 0);

      /// Add data to cache, returns number of added vectors
      void AddData(const Matrix<BaseFloat>& rFeatures, const Matrix<BaseFloat>& rDesired);
      /// Randomizes the cache
      void Randomize();
      /// Get the bunch of training data
      void GetBunch(Matrix<BaseFloat>& rFeatures, Matrix<BaseFloat>& rDesired);


      /// Returns true if the cache was completely filled
      bool Full()
      { return (mState == FULL); }
      
      /// Returns true if the cache is empty
      bool Empty()
      { return (mState == EMPTY || mIntakePos < mBunchsize); }
      
      /// Number of discarded frames
      int Discarded() 
      { return mDiscarded; }
      
      /// Set the trace message level
      void Trace(int trace)
      { mTrace = trace; }

    private:
    
      static long int GenerateRandom(int max)
      { return lrand48() % max; }
      
      State mState; ///< Current state of the cache

      size_t mIntakePos; ///< Number of intaken vectors by AddData
      size_t mExhaustPos; ///< Number of exhausted vectors by GetBunch
      
      size_t mCachesize; ///< Size of cache
      size_t mBunchsize; ///< Size of bunch
      int mDiscarded; ///< Number of discarded frames

      Matrix<BaseFloat> mFeatures; ///< Feature cache
      Matrix<BaseFloat> mFeaturesRandom; ///< Feature cache
      Matrix<BaseFloat> mFeaturesLeftover; ///< Feature cache
      
      Matrix<BaseFloat> mDesired;  ///< Desired vector cache
      Matrix<BaseFloat> mDesiredRandom;  ///< Desired vector cache
      Matrix<BaseFloat> mDesiredLeftover;  ///< Desired vector cache

      bool mRandomized;

      int mTrace;
  }; 

}

#endif
