/*
 * barrier.h
 *
 * This header file describes the "barrier" synchronization
 * construct. The type barrier_t describes the full state of the
 * barrier including the POSIX 1003.1c synchronization objects
 * necessary.
 *
 * A barrier causes threads to wait until a set of threads has
 * all "reached" the barrier. The number of threads required is
 * set when the barrier is initialized, and cannot be changed
 * except by reinitializing.
 */
#include <pthread.h>

#ifndef barrier_h
#define barrier_h

namespace TNet {

/*
 * Structure describing a barrier.
 */
class Barrier {
 public:
  Barrier(int count=0);
  ~Barrier();
  void SetThreshold(int thr);
  int Wait();
 private:
  pthread_mutex_t     mutex_;          /* Control access to barrier */
  pthread_cond_t      cv_;             /* wait for barrier */
  int                 threshold_;      /* number of threads required */
  int                 counter_;        /* current number of threads */
  int                 cycle_;          /* alternate wait cycles (0 or 1) */
};

}//namespace TNet

#endif

