#ifndef _CUCOMMON_H_
#define _CUCOMMON_H_

#include <iostream>
#include <sstream>

#include <cuda_runtime_api.h>

#include "Error.h"



#define cuSafeCall(fun) \
{ \
  int ret; \
  if((ret = (fun)) != 0) { \
    std::ostringstream os; \
    os << "CUDA ERROR #" << ret << " " << __FILE__ ":" << __LINE__ << " "  << __func__ << "()" << " '" << #fun << "' " << cudaGetErrorString((cudaError_t)ret); \
    throw(MyException(os.str())); \
  } \
  cudaThreadSynchronize(); \
} 




namespace TNet {

  /** The size of edge of CUDA square block **/
  static const int CUBLOCK = 16;

  /** Number of blocks in which is split task of size 'size' **/
  inline int n_blocks(int size, int block_size) 
  { return size / block_size + ((size % block_size == 0)? 0 : 1); }

  /** Printing dim3 output operator **/
  inline std::ostream& operator<<(std::ostream& os, dim3 arr) {
    os << "[" << arr.x << "," << arr.y << "," << arr.z << "]";
    return os;
  }

}



#endif
