

#include "cuCache.h"
#include "cumath.h"



namespace TNet {

  CuCache::
  CuCache()
    : mState(EMPTY), mIntakePos(0), mExhaustPos(0), mDiscarded(0), 
      mRandomized(false), mTrace(0)
  { }

  CuCache::
  ~CuCache()
  { }

  void
  CuCache::
  Init(size_t cachesize, size_t bunchsize)
  {
    if((cachesize % bunchsize) != 0) {
      Error("Non divisible cachesize by bunchsize");
    }
    
    mCachesize = cachesize;
    mBunchsize = bunchsize;

    mState = EMPTY;

    mIntakePos = 0;
    mExhaustPos = 0;

    mRandomized = false;

  }

  void 
  CuCache::
  AddData(const CuMatrix<BaseFloat>& rFeatures, const CuMatrix<BaseFloat>& rDesired)
  {
    assert(rFeatures.Rows() == rDesired.Rows());

    //lazy buffers allocation
    if(mFeatures.Rows() != mCachesize) {
      mFeatures.Init(mCachesize,rFeatures.Cols());
      mDesired.Init(mCachesize,rDesired.Cols());
    }

    //warn if segment longer than half-cache
    if(rFeatures.Rows() > mCachesize/2) {
      std::ostringstream os;
      os << "Too long segment and small feature cache! "
         << " cachesize: " << mCachesize
         << " segmentsize: " << rFeatures.Rows();
      Warning(os.str());
    }

    //change state
    if(mState == EMPTY) { 
      if(mTrace&3) std::cout << "/" << std::flush; 
      mState = INTAKE; mIntakePos = 0;
     
      //check for leftover from previous segment 
      int leftover = mFeaturesLeftover.Rows();
      //check if leftover is not bigger than cachesize
      if(leftover > mCachesize) {
        std::ostringstream os;
        os << "Too small feature cache: " << mCachesize
           << ", truncating: "
           << leftover - mCachesize << " frames from previous segment leftover";
        //Error(os.str());
        Warning(os.str());
        leftover = mCachesize;
      }
      //prefill cache with leftover
      if(leftover > 0) {
        mFeatures.CopyRows(leftover,0,mFeaturesLeftover,0);
        mDesired.CopyRows(leftover,0,mDesiredLeftover,0);
        mFeaturesLeftover.Destroy();
        mDesiredLeftover.Destroy();
        mIntakePos += leftover;
      } 
    }

    assert(mState == INTAKE);
    assert(rFeatures.Rows() == rDesired.Rows());
    if(mTrace&2) std::cout << "F" << std::flush; 

    int cache_space = mCachesize - mIntakePos;
    int feature_length = rFeatures.Rows();
    int fill_rows = (cache_space<feature_length)? cache_space : feature_length;
    int leftover = feature_length - fill_rows;

    assert(cache_space > 0);

    //copy the data to cache
    mFeatures.CopyRows(fill_rows,0,rFeatures,mIntakePos);
    mDesired.CopyRows(fill_rows,0,rDesired,mIntakePos);

    //copy leftovers
    if(leftover > 0) {
      mFeaturesLeftover.Init(leftover,mFeatures.Cols());
      mDesiredLeftover.Init(leftover,mDesired.Cols());
      mFeaturesLeftover.CopyRows(leftover,fill_rows,rFeatures,0);
      mDesiredLeftover.CopyRows(leftover,fill_rows,rDesired,0);
    }
 
    //update cursor
    mIntakePos += fill_rows;
    
    //change state
    if(mIntakePos == mCachesize) { 
      if(mTrace&3) std::cout << "\\" << std::flush; 
      mState = FULL;
    }
  }



  void
  CuCache::
  Randomize()
  {
    assert(mState == FULL || mState == INTAKE);

    if(mTrace&3) std::cout << "R" << std::flush;

    //lazy initialization of hte output buffers
    mFeaturesRandom.Init(mCachesize,mFeatures.Cols());
    mDesiredRandom.Init(mCachesize,mDesired.Cols());

    //generate random series of integers
    Vector<int> randmask(mIntakePos);
    for(unsigned int i=0; i<mIntakePos; i++) {
      randmask[i]=i;
    }
    int* ptr = randmask.pData();
    std::random_shuffle(ptr, ptr+mIntakePos, GenerateRandom);

    CuVector<int> cu_randmask;
    cu_randmask.CopyFrom(randmask);

    //randomize
    CuMath<BaseFloat>::Randomize(mFeaturesRandom,mFeatures,cu_randmask);
    CuMath<BaseFloat>::Randomize(mDesiredRandom,mDesired,cu_randmask);

    mRandomized = true;

  }

  void
  CuCache::
  GetBunch(CuMatrix<BaseFloat>& rFeatures, CuMatrix<BaseFloat>& rDesired)
  {
    if(mState == EMPTY) {
      Error("GetBunch on empty cache!!!");
    }

    //change state if full...
    if(mState == FULL) { 
      if(mTrace&3) std::cout << "\\" << std::flush; 
      mState = EXHAUST; mExhaustPos = 0; 
    }

    //final cache is not completely filled
    if(mState == INTAKE) //&& mpFeatures->EndOfList()
    { 
      if(mTrace&3) std::cout << "\\-LAST\n" << std::flush; 
      mState = EXHAUST; mExhaustPos = 0; 
    } 

    assert(mState == EXHAUST);

    //init the output
    rFeatures.Init(mBunchsize,mFeatures.Cols());
    rDesired.Init(mBunchsize,mDesired.Cols());

    //copy the output
    if(mRandomized) {
      rFeatures.CopyRows(mBunchsize,mExhaustPos,mFeaturesRandom,0);
      rDesired.CopyRows(mBunchsize,mExhaustPos,mDesiredRandom,0);
    } else {
      rFeatures.CopyRows(mBunchsize,mExhaustPos,mFeatures,0);
      rDesired.CopyRows(mBunchsize,mExhaustPos,mDesired,0);
    }

    //update cursor
    mExhaustPos += mBunchsize;

    //change state to EMPTY
    if(mExhaustPos > mIntakePos-mBunchsize) {
      //we don't have more complete bunches...
      mDiscarded += mIntakePos - mExhaustPos;

      mState = EMPTY;
    }
  }


}
