#ifndef _CUDEVICE_H_
#define _CUDEVICE_H_

#include <map>
#include <string>
#include <iostream>

namespace TNet {

  /**
   * Singleton object which represents CUDA device
   * responsible for CUBLAS initilalisation
   * and memory block registration
   */
  class CuDevice 
  {
    // Singleton interface...
    private:
      CuDevice();
      CuDevice(CuDevice&);
      CuDevice& operator=(CuDevice&);

    public:
      ~CuDevice();
      static CuDevice& Instantiate()
      { return msDevice; }

    private:
      static CuDevice msDevice;


    /**********************************/
    // Instance interface
    public:

      void SelectGPU(int gpu_id);
     
      /// Check if the CUDA device is in the system      
      bool IsPresent()
      { return mIsPresent; }

      void Verbose(bool verbose)
      { mVerbose = verbose; }

      /// Sum the IO time
      void AccuProfile(const std::string& key,double time) 
      { 
        if(mProfileMap.find(key) == mProfileMap.end()) {
          mProfileMap[key] = 0.0;
        }
        mProfileMap[key] += time;
      }

      void PrintProfile()
      { 
        std::cout << "[cudevice profile]\n";
        std::map<std::string, double>::iterator it;
        for(it = mProfileMap.begin(); it != mProfileMap.end(); ++it) {
          std::cout << it->first << "\t" << it->second << "s\n";
        }
      }

      void ResetProfile()
      { mProfileMap.clear(); }

      std::string GetFreeMemory();


    private:
      std::map<std::string, double> mProfileMap;
      bool mIsPresent;
      bool mVerbose;
  }; //class CuDevice


}


#endif
