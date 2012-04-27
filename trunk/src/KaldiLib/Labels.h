#ifndef _LABELS_H_
#define _LABELS_H_


#include "Matrix.h"
#include "MlfStream.h"
#include "Features.h"

#include <map>
#include <iostream>

namespace TNet {


  class FeaCatPool;

  /**
   * Desired matrix generation object,
   * supports background-reading and caching, however can be 
   * used in foreground as well by GenDesiredMatrix()
   */
  class LabelRepository 
  {
    typedef std::map<std::string,size_t> TagToIdMap;

    public:
      LabelRepository()
        : _mpLabelStream(NULL), mpLabelStream(NULL), mpLabelDir(NULL), mpLabelExt(NULL), mGenDesiredMatrixTime(0), mIndexTime(0), mTrace(0) 
      { }

      ~LabelRepository()
      { 
        if(mTrace&4) {
          std::cout << "[LabelRepository -- indexing:" << mIndexTime << "s"
                       " genDesiredMatrix:" << mGenDesiredMatrixTime << "s]" << std::endl;
        }
        delete mpLabelStream;
        delete _mpLabelStream;
      }

      /// Initialize the LabelRepository      
      void Init(const char* pLabelMlfFile, const char* pOutputLabelMapFile, const char* pLabelDir, const char* pLabelExt);

      /// Set trace level
      void Trace(int trace)
      { mTrace = trace; }

      /// Get desired matrix from labels
      void GenDesiredMatrix(BfMatrix& rDesired, size_t nFrames, size_t sourceRate, const char* pFeatureLogical);

    private:
      /// Prepare the state-label to state-id map
      void ReadOutputLabelMap(const char* file);
      
    private:
      // Streams and state-map
      std::ifstream* _mpLabelStream; ///< Helper stream for Label stream
      IMlfStream* mpLabelStream;     ///< Label stream
      std::istringstream mGenDesiredMatrixStream; ///< Label file parsing stream
     
      const char* mpLabelDir;  ///< Label dir in MLF 
      const char* mpLabelExt;  ///< Label ext in MLF
      char mpLabelFile[4096];  ///< Buffer for filenames in MLF
      
      TagToIdMap mLabelMap; ///< Map of state tags to net output indices

      double mGenDesiredMatrixTime;
      float  mIndexTime;

      int mTrace;
  };

}//namespace

#endif
