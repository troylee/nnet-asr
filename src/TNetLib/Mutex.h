
#include <pthread.h>

namespace TNet {

/**
 * This class encapsulates mutex to ensure 
 * exclusive access to some critical section
 * which manipulates shared resources.
 *
 * The mutex must be unlocked from the 
 * SAME THREAD which locked it
 */
class Mutex {
 public:
  Mutex();
  ~Mutex();

  void Lock();

  /**
   * Try to lock the mutex without waiting for it.
   * Returns: true when lock successfull,
   *         false when mutex was already locked
   */
  bool TryLock();

  void Unlock();

 private:
  pthread_mutex_t mutex_;
};

} //namespace TNet
