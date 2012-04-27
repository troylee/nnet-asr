
#include <pthread.h>
#include <cerrno>

#include "Error.h"
#include "Mutex.h"

namespace TNet {
  

Mutex::Mutex() {
  if(0 != pthread_mutex_init(&mutex_,NULL)) 
    KALDI_ERR << "Cannot initialize mutex";
}


Mutex::~Mutex() {
  if(0 != pthread_mutex_destroy(&mutex_)) 
    KALDI_ERR << "Cannot destroy mutex";
}


void Mutex::Lock() {
  if(0 != pthread_mutex_lock(&mutex_))
    KALDI_ERR << "Error on locking mutex";
}

 
bool Mutex::TryLock() {
  int ret = pthread_mutex_lock(&mutex_);
  switch (ret) {
    case 0: return true;
    case EBUSY: return false;
    default: KALDI_ERR << "Error on try-locking mutex";
  }
  return 0;//make compiler not complain
}


void Mutex::Unlock() {
  if(0 != pthread_mutex_unlock(&mutex_))
    KALDI_ERR << "Error on unlocking mutex";
}


  
}//namespace TNet

