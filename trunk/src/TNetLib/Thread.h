#ifndef _TNET_THREAD_H
#define _TNET_THREAD_H

namespace TNet {

class Thread {
 public:
  Thread() 
  { }
  virtual ~Thread() 
  { }

  int Start(void* arg);

 protected:
  static void* EntryPoint(void*);
  virtual void Execute(void*) = 0; ///< Override this function
  void* Arg() const { return arg_; }
  void Arg(void* a) { arg_ = a; }

 private:
  pthread_t thread_id_;
  void * arg_;
};

int Thread::Start(void * arg) {
  Arg(arg); // store user data
 
  int ret=0;
  //create thread as detached (don't wait for it)
  pthread_attr_t tattr;
  ret |= pthread_attr_init(&tattr);
  ret |= pthread_attr_setdetachstate(&tattr,PTHREAD_CREATE_DETACHED);
  ret |= pthread_create(&thread_id_, &tattr, &Thread::EntryPoint, this);
  if(ret != 0) KALDI_ERR << "Failed to create thread";
  return ret;
}

/*static */
void* Thread::EntryPoint(void* pthis) try {
  Thread* pt = (Thread*)pthis;
  pt->Execute(pt->Arg());
  return NULL;
} catch (std::exception& rExc) {
  std::cerr << "Exception thrown" << std::endl;
  std::cerr << rExc.what() << std::endl;
  exit(1);
}


} //namespace TNet

#endif
