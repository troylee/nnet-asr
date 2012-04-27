
#include "Semaphore.h"

namespace TNet {
  
  Semaphore::
  Semaphore(int initValue) 
  {
    mSemValue = initValue;
    pthread_mutex_init(&mMutex, NULL);
    pthread_cond_init(&mCond, NULL);
  }

  Semaphore::
  ~Semaphore()
  {
    pthread_mutex_destroy(&mMutex);
    pthread_cond_destroy(&mCond);
  }

  int 
  Semaphore::
  TryWait()
  {
    pthread_mutex_lock(&mMutex);
    if(mSemValue > 0) {
      mSemValue--;
      pthread_mutex_unlock(&mMutex);
      return 0;
    }
    pthread_mutex_unlock(&mMutex);
    return -1;
  }

  void 
  Semaphore::
  Wait()
  {
    pthread_mutex_lock(&mMutex);
    while(mSemValue <= 0) {
      pthread_cond_wait(&mCond, &mMutex);
    }
    mSemValue--;
    pthread_mutex_unlock(&mMutex);
  }

  void
  Semaphore::
  Post()
  {
    pthread_mutex_lock(&mMutex);
    mSemValue++;
    pthread_cond_signal(&mCond);
    pthread_mutex_unlock(&mMutex);
  }

  int
  Semaphore::
  GetValue()
  { return mSemValue; }



} //namespace
