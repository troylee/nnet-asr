
#include <cudevice.h>
#include <cublas.h>
#include <cuda.h>

///////////////////
//DEBUG: Just make sure it compiles...
#include "cumatrix.h"
#include "cuvector.h"
#include "cumath.h"
template class TNet::CuMatrix<float>;
template class TNet::CuVector<float>;
template class TNet::CuMath<float>;
///////////////////

namespace TNet {


  /**********************************************************************************
   * CuDevice::
   */
  CuDevice::
  CuDevice()
    : mIsPresent(false), mVerbose(false)
  {
    //get number of devices
    int N_GPU = 0;
    cudaGetDeviceCount(&N_GPU);

    //select device if more than one
    if(N_GPU > 1) {
      char name[128];
      size_t free, total;
      std::vector<float> free_mem_ratio;
      //get ratios of memory use
      std::cout << "Selecting from " << N_GPU << " GPUs\n";
      for(int n=0; n<N_GPU; n++) {
        std::cout << "cudaSetDevice(" << n << "): ";
        cuSafeCall(cudaSetDevice(n));//context created by cuSafeCall(...)
        cuDeviceGetName(name,128,n);
        std::cout << name << "\t";
        cuSafeCall(cuMemGetInfo(&free,&total));
        std::cout << "free: " << free/1024/1024 << "M, "
                  << "total: "<< total/1024/1024 << "M, "
                  << "ratio: "<< free/(float)total << "\n";
        free_mem_ratio.push_back(free/(float)total);
        cudaThreadExit();//destroy context
      }
      //find GPU with max free memory
      int max_id=0;
      for(int n=1; n<free_mem_ratio.size(); n++) {
        if(free_mem_ratio[n] > free_mem_ratio[max_id]) max_id=n;
      }
      std::cout << "Selected device: " << max_id << " (automatically)\n";
      cuSafeCall(cudaSetDevice(max_id));
    }
      
    if(N_GPU > 0) {
      //initialize the CUBLAS
      cuSafeCall(cublasInit());
      mIsPresent = true;
    } else {
      Warning("No CUDA enabled GPU is present!");
    }
  }

  CuDevice::
  ~CuDevice()
  {
    if(mIsPresent) {
      cuSafeCall(cublasShutdown());
      if(mVerbose) {
        TraceLog("CUBLAS released");
        PrintProfile();
      }
    } else {
      Warning("No CUDA enabled GPU was present!");
    }
  }


  void 
  CuDevice::
  SelectGPU(int gpu_id)
  {
    //get number of devices
    int N_GPU = 0;
    cudaGetDeviceCount(&N_GPU);
    if(gpu_id >= N_GPU) {
      KALDI_ERR << "Cannot select GPU " << gpu_id 
                << ", detected " << N_GPU << " CUDA capable cards!";
    }
    //release old card
    cuSafeCall(cublasShutdown());
    cudaThreadExit();
    //select new card
    cuSafeCall(cudaSetDevice(gpu_id));
    //initialize CUBLAS
    cuSafeCall(cublasInit());
    std::cout << "Selected device " << gpu_id << " (manually)\n";
  }


  std::string
  CuDevice::
  GetFreeMemory()
  {
    size_t mem_free, mem_total;
    cuMemGetInfo(&mem_free, &mem_total);
    std::ostringstream os;
    os << "Free:" << mem_free/(1024*1024) << "MB "
       << "Used:" << (mem_total-mem_free)/(1024*1024) << "MB "
       << "Total:" << mem_total/(1024*1024) << "MB";
    return os.str();
  }


  ////////////////////////////////////////////////
  // Instance of the static singleton 
  //
  CuDevice CuDevice::msDevice;
  //
  ////////////////////////////////////////////////
  


}


