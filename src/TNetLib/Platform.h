#ifndef _TNET_PLATFORM_H
#define _TNET_PLATFORM_H

#include "Thread.h"
#include "Matrix.h"

#include "Features.h"
#include "Labels.h"

#include "Cache.h"
#include "Nnet.h"
#include "ObjFun.h"

#include "Mutex.h"
#include "Semaphore.h"
#include "Barrier.h"
#include "Thread.h"

#include <vector>
#include <list>
#include <iterator>

namespace TNet {

class PlatformThread;

class Platform {

/*
* Variables to be initialized directly from the main function
*/
public:
  FeatureRepository feature_;
  LabelRepository label_;

  Network nnet_transf_;
  Network nnet_;
  ObjectiveFunction* obj_fun_;

  int bunchsize_;
  int cachesize_;
  bool randomize_;
   
  int start_frm_ext_;
  int end_frm_ext_;

  int trace_;
  bool crossval_;
  
  long int seed_;

 /*
  * Variables to be used internally during the multi-threaded training
  */
 private:
  Semaphore semaphore_read_;
 
  std::vector<std::list<Matrix<BaseFloat>*> > feature_buf_;
  std::vector<std::list<Matrix<BaseFloat>*> > label_buf_;
  std::vector<Mutex> mutex_buf_;

  std::vector<Network*> nnet_transf2_;

  std::vector<Cache> cache_;

  std::vector<Network*> nnet2_;
  std::vector<ObjectiveFunction*> obj_fun2_;
  std::vector<bool> sync_mask_;

  Barrier barrier_;
  bool end_reading_;
  std::vector<Timer> tim_;
  std::vector<double> tim_accu_;

  int num_thr_;
  Semaphore semaphore_endtrain_;
  Semaphore semaphore_endtrain2_;

 public:
  Mutex cout_mutex_;

 /*
  * Methods
  */
 public:
  Platform()
   : bunchsize_(0), cachesize_(0), randomize_(false),
     start_frm_ext_(0), end_frm_ext_(0), trace_(0),
     crossval_(false), seed_(0),
     end_reading_(false), num_thr_(0)
  { }

  ~Platform()
  {
    for(size_t i=0; i<nnet_transf2_.size(); i++) {
      delete nnet_transf2_[i];
    }
    for(size_t i=0; i<nnet2_.size(); i++) {
      delete nnet2_[i];
    }
    for(size_t i=0; i<obj_fun2_.size(); i++) {
      delete obj_fun2_[i];
    }
  }
 
  /// Run the training using num_threads threads
  void RunTrain(int num_threads);

 private:
  /// The data-reading thread
  void ReadData();
  /// The training thread
  void Thread(int thr);

 friend class PlatformThread;
};



/**
 * Inherit Thread for the training threads
 */
class PlatformThread : public Thread {
 public:
  PlatformThread(Platform* pf)
   : platform_(*pf)
  { }
 
 private:
  void Execute(void* arg) {
    long long thr_id = reinterpret_cast<long long>(arg);
    platform_.Thread(thr_id);
  }
   
 private:
  Platform& platform_;
};





void Platform::RunTrain(int num_thr) {
  num_thr_ = num_thr;
  
  /*
   * Initialize parallel training
   */
  feature_buf_.resize(num_thr);
  label_buf_.resize(num_thr);
  mutex_buf_.resize(num_thr);
  cache_.resize(num_thr);
  sync_mask_.resize(num_thr);
  barrier_.SetThreshold(num_thr);

  tim_.resize(num_thr);
  tim_accu_.resize(num_thr,0.0);

  int bunchsize = bunchsize_/num_thr;
  int cachesize = (cachesize_/num_thr/bunchsize)*bunchsize;
  std::cout << "Bunchsize:" << bunchsize << "*" << num_thr << "=" << bunchsize*num_thr
            << " Cachesize:" << cachesize << "*" << num_thr << "=" << cachesize*num_thr << "\n";
  for(int i=0; i<num_thr; i++) {
    //clone transforms
    nnet_transf2_.push_back(nnet_transf_.Clone()); 
    //create cache
    cache_[i].Init(cachesize,bunchsize,seed_);
    cache_[i].Trace(trace_);
    //clone networks
    nnet2_.push_back(nnet_.Clone());
    //clone objective function objects
    obj_fun2_.push_back(obj_fun_->Clone());
    //enable threads to sync weights
    sync_mask_[i] = true;
  }

  /*
   * Run training threads
   */
  std::vector<PlatformThread*> threads;
  for(intptr_t i=0; i<num_thr; i++) {
    PlatformThread* t = new PlatformThread(this);
    t->Start(reinterpret_cast<void*>(i));
    threads.push_back(t);
  }

  /*
   * Read the training data
   */
  ReadData();

  /*
   * Wait for training to finish
   */
  semaphore_endtrain2_.Wait(); 

}



void Platform::ReadData() try {
  cout_mutex_.Lock();  
  std::cout << "queuesize " << feature_.QueueSize() << "\n";
  cout_mutex_.Unlock();  

  int thr = 0;
  for(feature_.Rewind();!feature_.EndOfList();feature_.MoveNext()) {
    Matrix<BaseFloat>* fea = new Matrix<BaseFloat>;
    Matrix<BaseFloat>* lab = new Matrix<BaseFloat>;

    feature_.ReadFullMatrix(*fea);
    label_.GenDesiredMatrix(*lab,
                            fea->Rows()-start_frm_ext_-end_frm_ext_,
                            feature_.CurrentHeader().mSamplePeriod,
                            feature_.Current().Logical().c_str());

    
    fea->CheckData(feature_.Current().Logical());

    mutex_buf_[thr].Lock();
    feature_buf_[thr].push_back(fea);
    label_buf_[thr].push_back(lab);
    mutex_buf_[thr].Unlock();

    //suspend reading when shortest buffer has 50 matrices
    if(thr == 0) {
      int minsize=1e6;
      for(size_t i=0; i<feature_buf_.size(); i++) {
        int s = feature_buf_[i].size();
        if(s < minsize) minsize = s;
      }
      if(minsize > 20) semaphore_read_.Wait();
    }

    thr = (thr+1) % num_thr_;
  }

  std::cout << "[Reading finished]\n" << std::flush; 
  end_reading_ = true;

} catch (std::exception& rExc) {
  std::cerr << "Exception thrown" << std::endl;
  std::cerr << rExc.what() << std::endl;
  exit(1);
}

void Platform::Thread(int thr_id) try {

  const int thr = thr_id; //make id const for safety!

  while(1) {
    //fill the cache
    while(!cache_[thr].Full() && !(end_reading_ && (feature_buf_[thr].size() == 0))) {

      if(feature_buf_[thr].size() <= 5) {
        semaphore_read_.Post();//wake the reader
      }
      if(feature_buf_[thr].size() == 0) {
        cout_mutex_.Lock();  
        std::cout << "Thread" << thr << ",waiting for data\n";
        cout_mutex_.Unlock();  
        sleep(1);
      } else {
        //get the matrices
        mutex_buf_[thr].Lock();
        Matrix<BaseFloat>* fea = feature_buf_[thr].front();
        Matrix<BaseFloat>* lab = label_buf_[thr].front();
        feature_buf_[thr].pop_front();
        label_buf_[thr].pop_front();
        mutex_buf_[thr].Unlock();

        //transform the features
        Matrix<BaseFloat> fea_transf;
        nnet_transf2_[thr]->Propagate(*fea,fea_transf);

        //trim the ext
        SubMatrix<BaseFloat> fea_trim(
          fea_transf,
          start_frm_ext_,
          fea_transf.Rows()-start_frm_ext_-end_frm_ext_,
          0,
          fea_transf.Cols()
        );

        //add to cache
        cache_[thr].AddData(fea_trim,*lab);

        delete fea; delete lab;
      }
    }

    //no more data, end training...
    if(cache_[thr].Empty()) break;

    if(randomize_) { cache_[thr].Randomize(); }


    //std::cout << "Thread" << thr << ", Cache#" << nr_cache++ << "\n";

    //train from cache
    Matrix<BaseFloat> fea2,lab2,out,err;
    while(!cache_[thr].Empty()) {
      cache_[thr].GetBunch(fea2,lab2);
      nnet2_[thr]->Propagate(fea2,out);
      obj_fun2_[thr]->Evaluate(out,lab2,&err);

      if(!crossval_) {
        nnet2_[thr]->Backpropagate(err);

         tim_[thr].Start();
        barrier_.Wait();//*********/
         tim_[thr].End(); tim_accu_[thr] += tim_[thr].Val();
       
        //sum the gradient and bunchsize
        for(int i=0; i<num_thr_; i++) {
          if(sync_mask_[i]) {
            nnet_.AccuGradient(*nnet2_[i],thr,num_thr_);
            if(thr == 0) nnet_.AccuBunchsize(*nnet2_[i]);
          }
        }

         tim_[thr].Start();
        barrier_.Wait();//*********/
         tim_[thr].End(); tim_accu_[thr] += tim_[thr].Val();

        //update
        nnet_.Update(thr,num_thr_);
       
         tim_[thr].Start();
        barrier_.Wait();//*********/
         tim_[thr].End(); tim_accu_[thr] += tim_[thr].Val();

        //reset the bunchsize counter
        if(thr == 0) nnet_.ResetBunchsize();
      }
    }

  }

  std::cout << "Thread" << thr << " end of data\n";
  
  //deactivate threads' update from summing
  sync_mask_[thr] = false;
  //increase number of finished threads
  semaphore_endtrain_.Post();
   
  //synchronize the updates of other threads
  while(1) {
    barrier_.Wait();//*********/
    if(semaphore_endtrain_.GetValue() == num_thr_) break;
        
    //sum the gradient and bunchsize
    for(int i=0; i<num_thr_; i++) {
      if(sync_mask_[i]) {
        nnet_.AccuGradient(*nnet2_[i],thr,num_thr_);
        if(thr == 0) nnet_.AccuBunchsize(*nnet2_[i]);
      }
    }
    barrier_.Wait();//*********/
    //update
    nnet_.Update(thr,num_thr_);
    barrier_.Wait();//*********/
    //reset bunchsize counter
    if(thr == 0) nnet_.ResetBunchsize();
  }

  //finally merge objfun stats
  if(thr == 0) {
    for(int i=0; i<num_thr_; i++) {
      obj_fun_->MergeStats(*obj_fun2_[i]);
    }
    
    cout_mutex_.Lock();
    std::cout << "Barrier waiting times per thread\n"; 
    std::copy(tim_accu_.begin(),tim_accu_.end(),std::ostream_iterator<double>(std::cout," "));
    std::cout << "\n";
    cout_mutex_.Unlock();
  }

  cout_mutex_.Lock();
  std::cout << "[Thread" << thr << " finished]\n";
  cout_mutex_.Unlock();

  if(thr == 0) {
    semaphore_endtrain2_.Post();
  }
} catch (std::exception& rExc) {
  std::cerr << "Exception thrown" << std::endl;
  std::cerr << rExc.what() << std::endl;
  exit(1);
}



}//namespace TNet

#endif
