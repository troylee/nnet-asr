/*
 * barrier.c
 *
 * This file implements the "barrier" synchronization construct.
 *
 * A barrier causes threads to wait until a set of threads has
 * all "reached" the barrier. The number of threads required is
 * set when the barrier is initialized, and cannot be changed
 * except by reinitializing.
 *
 * The barrier_init() and barrier_destroy() functions,
 * respectively, allow you to initialize and destroy the
 * barrier.
 *
 * The barrier_wait() function allows a thread to wait for a
 * barrier to be completed. One thread (the one that happens to
 * arrive last) will return from barrier_wait() with the status
 * -1 on success -- others will return with 0. The special
 * status makes it easy for the calling code to cause one thread
 * to do something in a serial region before entering another
 * parallel section of code.
 */
#include <pthread.h>
#include "Error.h"
#include "Barrier.h"

namespace TNet {

/*
 * Initialize a barrier for use.
 */
Barrier::Barrier(int count)
 : threshold_(count), counter_(count), cycle_(0) {

  if(0 != pthread_mutex_init(&mutex_, NULL))
    KALDI_ERR << "Cannot initialize mutex";
  
  if(0 != pthread_cond_init(&cv_, NULL)) {
    pthread_mutex_destroy(&mutex_);
    KALDI_ERR << "Cannot initilize condv";
  }
}

/*
 * Destroy a barrier when done using it.
 */
Barrier::~Barrier() {

  if(0 != pthread_mutex_lock(&mutex_))
    KALDI_ERR << "Cannot lock mutex";

  /*
   * Check whether any threads are known to be waiting; report
   * "BUSY" if so.
   */
  if(counter_ != threshold_) {
    pthread_mutex_unlock (&mutex_);
    KALDI_ERR << "Cannot destroy barrier with waiting thread";
  }

  if(0 != pthread_mutex_unlock(&mutex_))
    KALDI_ERR << "Cannot unlock barrier";

  /*
   * If unable to destroy either 1003.1c synchronization
   * object, halt
   */
  if(0 != pthread_mutex_destroy(&mutex_))
    KALDI_ERR << "Cannot destroy mutex";

  if(0 != pthread_cond_destroy(&cv_)) 
    KALDI_ERR << "Cannot destroy condv";
}


void Barrier::SetThreshold(int thr) {
  if(counter_ != threshold_) 
    KALDI_ERR << "Cannot set threshold, while a thread is waiting";

  threshold_ = thr; counter_ = thr;
}



/*
 * Wait for all members of a barrier to reach the barrier. When
 * the count (of remaining members) reaches 0, broadcast to wake
 * all threads waiting.
 */
int Barrier::Wait() {
  int status, cancel, tmp, cycle;

  if(threshold_ == 0)
    KALDI_ERR << "Cannot wait when Threshold value was not set";

  if(0 != pthread_mutex_lock(&mutex_)) 
    KALDI_ERR << "Cannot lock mutex";

  cycle = cycle_;   /* Remember which cycle we're on */

  if(--counter_ == 0) {
    cycle_ = !cycle_;
    counter_ = threshold_;
    status = pthread_cond_broadcast(&cv_);
    /*
     * The last thread into the barrier will return status
     * -1 rather than 0, so that it can be used to perform
     * some special serial code following the barrier.
     */
    if(status == 0) status = -1;
  } else {
    /*
     * Wait with cancellation disabled, because barrier_wait
     * should not be a cancellation point.
     */
    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &cancel);

    /*
     * Wait until the barrier's cycle changes, which means
     * that it has been broadcast, and we don't want to wait
     * anymore.
     */
    while (cycle == cycle_) {
      status = pthread_cond_wait(&cv_, &mutex_);
      if (status != 0) break;
    }

    pthread_setcancelstate(cancel, &tmp);
  }
  /*
   * Ignore an error in unlocking. It shouldn't happen, and
   * reporting it here would be misleading -- the barrier wait
   * completed, after all, whereas returning, for example,
   * EINVAL would imply the wait had failed. The next attempt
   * to use the barrier *will* return an error, or hang, due
   * to whatever happened to the mutex.
   */
  pthread_mutex_unlock (&mutex_);
  return status;          /* error, -1 for waker, or 0 */
}


}//namespace TNet
