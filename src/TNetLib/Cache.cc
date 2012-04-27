
#include <sys/time.h>

#include "Cache.h"
#include "Matrix.h"
#include "Vector.h"


namespace TNet {

  Cache::
  Cache()
    : mState(EMPTY), mIntakePos(0), mExhaustPos(0), mDiscarded(0), 
      mRandomized(false), mTrace(0)
  { }

  Cache::
  ~Cache()
  { }

  void
  Cache::
  Init(size_t cachesize, size_t bunchsize, long int seed)
  {
    if((cachesize % bunchsize) != 0) {
      KALDI_ERR << "Non divisible cachesize" << cachesize
                << " by bunchsize" << bunchsize;
    }
    
    mCachesize = cachesize;
    mBunchsize = bunchsize;

    mState = EMPTY;

    mIntakePos = 0;
    mExhaustPos = 0;

    mRandomized = false;

    if(seed == 0) {
      //generate seed
      struct timeval tv;
      if (gettimeofday(&tv, 0) == -1) {
        Error("gettimeofday does not work.");
        exit(-1);
      }
      seed = (int)(tv.tv_sec) + (int)tv.tv_usec + (int)(tv.tv_usec*tv.tv_usec);
    }

    srand48(seed);

  }

  void 
  Cache::
  AddData(const Matrix<BaseFloat>& rFeatures, const Matrix<BaseFloat>& rDesired)
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
        memcpy(mFeatures.pData(),mFeaturesLeftover.pData(),
          (mFeaturesLeftover.MSize() < mFeatures.MSize()?
           mFeaturesLeftover.MSize() : mFeatures.MSize()) 
        );
        memcpy(mDesired.pData(),mDesiredLeftover.pData(),
          (mDesiredLeftover.MSize() < mDesired.MSize()?
           mDesiredLeftover.MSize() : mDesired.MSize()) 
        );
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
    assert(mFeatures.Stride()==rFeatures.Stride());
    assert(mDesired.Stride()==rDesired.Stride());

    //copy the data to cache
    memcpy(mFeatures.pData()+mIntakePos*mFeatures.Stride(),
           rFeatures.pData(),
           fill_rows*mFeatures.Stride()*sizeof(BaseFloat));

    memcpy(mDesired.pData()+mIntakePos*mDesired.Stride(),
           rDesired.pData(),
           fill_rows*mDesired.Stride()*sizeof(BaseFloat));

    //copy leftovers
    if(leftover > 0) {
      mFeaturesLeftover.Init(leftover,mFeatures.Cols());
      mDesiredLeftover.Init(leftover,mDesired.Cols());

      memcpy(mFeaturesLeftover.pData(),
             rFeatures.pData()+fill_rows*rFeatures.Stride(),
             mFeaturesLeftover.MSize());

      memcpy(mDesiredLeftover.pData(),
             rDesired.pData()+fill_rows*rDesired.Stride(),
             mDesiredLeftover.MSize());       
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
  Cache::
  Randomize()
  {
    assert(mState == FULL || mState == INTAKE);

    if(mTrace&3) std::cout << "R" << std::flush;

    //lazy initialization of the output buffers
    mFeaturesRandom.Init(mCachesize,mFeatures.Cols());
    mDesiredRandom.Init(mCachesize,mDesired.Cols());

    //generate random series of integers
    Vector<int> randmask(mIntakePos);
    for(unsigned int i=0; i<mIntakePos; i++) {
      randmask[i]=i;
    }
    int* ptr = randmask.pData();
    std::random_shuffle(ptr, ptr+mIntakePos, GenerateRandom);

    //randomize
    for(int i=0; i<randmask.Dim(); i++) {
      mFeaturesRandom[i].Copy(mFeatures[randmask[i]]);
      mDesiredRandom[i].Copy(mDesired[randmask[i]]);
    }

    mRandomized = true;
  }

  void
  Cache::
  GetBunch(Matrix<BaseFloat>& rFeatures, Matrix<BaseFloat>& rDesired)
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
    if(mState == INTAKE) {
      if(mTrace&3) std::cout << "\\-LAST_CACHE\n" << std::flush; 
      mState = EXHAUST; mExhaustPos = 0; 
    } 

    assert(mState == EXHAUST);

    //init the output
    if(rFeatures.Rows()!=mBunchsize || rFeatures.Cols()!=mFeatures.Cols()) {
      rFeatures.Init(mBunchsize,mFeatures.Cols());
    }
    if(rDesired.Rows()!=mBunchsize || rDesired.Cols()!=mDesired.Cols()) {
      rDesired.Init(mBunchsize,mDesired.Cols());
    }

    //copy the output
    if(mRandomized) {
      memcpy(rFeatures.pData(),
             mFeaturesRandom.pData()+mExhaustPos*mFeatures.Stride(),
             rFeatures.MSize());

      memcpy(rDesired.pData(),
             mDesiredRandom.pData()+mExhaustPos*mDesired.Stride(),
             rDesired.MSize());
    } else {
      memcpy(rFeatures.pData(),
             mFeatures.pData()+mExhaustPos*mFeatures.Stride(),
             rFeatures.MSize());

      memcpy(rDesired.pData(),
             mDesired.pData()+mExhaustPos*mDesired.Stride(),
             rDesired.MSize());
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
