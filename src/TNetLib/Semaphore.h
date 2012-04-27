#ifndef _SEMPAHORE_H_
#define _SEMPAHORE_H_

#include <pthread.h>

namespace TNet {
  
  class Semaphore {
    public:
      Semaphore(int initValue = 0); 
      ~Semaphore();

      int TryWait();
      void Wait();
      void Post();
      int GetValue();

    private:
      int mSemValue;
      pthread_mutex_t mMutex;
      pthread_cond_t mCond;

  };
} //namespace

#endif
