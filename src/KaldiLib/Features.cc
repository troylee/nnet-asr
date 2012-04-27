
//enable feature repository profiling
#define PROFILING 1

#include <sstream>
#include <map>
#include <list>
#include <cstdio>

#include "Features.h"
#include "Tokenizer.h"
#include "StkMatch.h"
#include "Types.h"



namespace TNet
{
  const char 
  FeatureRepository::
  mpParmKindNames[13][16] =
  {
    {"WAVEFORM"},
    {"LPC"},
    {"LPREFC"},
    {"LPCEPSTRA"},
    {"LPDELCEP"},
    {"IREFC"},
    {"MFCC"},
    {"FBANK"},
    {"MELSPEC"},
    {"USER"},
    {"DISCRETE"},
    {"PLP"},
    {"ANON"}
  };

  //***************************************************************************
  //***************************************************************************

  FileListElem::
  FileListElem(const std::string & rFileName)
  {
    std::string::size_type  pos;
    
    mLogical = rFileName;
    mWeight  = 1.0;
    
    // some slash-backslash replacement hack
    for (size_t i = 0; i < mLogical.size(); i++) {
      if (mLogical[i] == '\\') {
        mLogical[i] = '/';
      }
    }
        
    // read sentence weight definition if any ( physical_file.fea[s,e]{weight} )
    if ((pos = mLogical.find('{')) != std::string::npos)
    {
      std::string       tmp_weight(mLogical.begin() + pos + 1, mLogical.end());
      std::stringstream tmp_ss(tmp_weight);

      tmp_ss >> mWeight;
      mLogical.erase(pos);
    }

    // look for "=" symbol and if found, split it
    if ((pos = mLogical.find('=')) != std::string::npos)
    {
      // copy all from mLogical[pos+1] till the end to mPhysical
      mPhysical.assign(mLogical.begin() + pos + 1, mLogical.end());
      // erase all from pos + 1 till the end from mLogical
      mLogical.erase(pos);
      // trim the leading and trailing spaces
      Trim(mPhysical);
      Trim(mLogical);
    }
    else
    {
      // trim the leading and trailing spaces
      Trim(mLogical);

      mPhysical = mLogical;
    }    
  }    


  //###########################################################################
  //###########################################################################
  // FeatureRepository section
  //###########################################################################
  //###########################################################################
  
  //***************************************************************************
  //***************************************************************************
  void 
  FeatureRepository::
  ReadCepsNormFile(
    const char *  pFileName, 
    char **       pLastFileName, 
    BaseFloat **      vec_buff,
    int           sampleKind, 
    CNFileType    type, 
    int           coefs)
  {
    FILE*   fp;
    int     i;
    char    s1[64];
    char    s2[64];
    const char*   typeStr = (type == CNF_Mean     ? "MEAN" :
                    type == CNF_Variance ? "VARIANCE" : "VARSCALE");
  
    const char*   typeStr2 = (type == CNF_Mean     ? "CMN" :
                    type == CNF_Variance ? "CVN" : "VarScale");
  
    if (*pLastFileName != NULL && !strcmp(*pLastFileName, pFileName)) {
      return;
    }
    free(*pLastFileName);
    *pLastFileName=strdup(pFileName);
    *vec_buff = (BaseFloat*) realloc(*vec_buff, coefs * sizeof(BaseFloat));
  
    if (*pLastFileName == NULL || *vec_buff== NULL) 
      throw std::runtime_error("Insufficient memory");
    
    if ((fp = fopen(pFileName, "r")) == NULL)  {
      throw std::runtime_error(std::string("Cannot open ") + typeStr2 
          + " pFileName: '" + pFileName + "'");
    }
    
    if ((type != CNF_VarScale
        && (fscanf(fp, " <%64[^>]> <%64[^>]>", s1, s2) != 2
          || strcmp(StrToUpper(s1), "CEPSNORM")
          || ReadParmKind(s2, false) != sampleKind))
        || fscanf(fp, " <%64[^>]> %d", s1, &i) != 2
        || strcmp(StrToUpper(s1), typeStr)
        || i != coefs) 
    {
      ParmKind2Str(sampleKind, s2);

      //std::cout << "[[[TADY!!!!]]]" << pFileName << "\n" << std::flush;

      throw std::runtime_error(std::string("")
            + (type == CNF_VarScale ? "" : "<CEPSNORM> <")
            + (type == CNF_VarScale ? "" : s2)
            + (type == CNF_VarScale ? "" : ">")
            + " <" + typeStr + " ... expected in " + typeStr2
            + " file " + pFileName);
    }
    
    for (i = 0; i < coefs; i++) {
      if (fscanf(fp, " "FLOAT_FMT, *vec_buff+i) != 1) {
        if (fscanf(fp, "%64s", s2) == 1) {
          throw std::runtime_error(std::string("Decimal number expected but '")
              + s2 + "' found in " + typeStr2 + " file " + pFileName);
        } 
        else if (feof(fp)) {
          throw std::runtime_error(std::string("Unexpected end of ") 
              + typeStr2 + " file "+ pFileName);
        } 
        else {
          throw std::runtime_error(std::string("Cannot read ") + typeStr2 
              + " file " + pFileName);
        }
      }
      
      if (type == CNF_Variance)      
        (*vec_buff)[i] = BaseFloat(1 / sqrt((*vec_buff)[i]));
      else if (type == CNF_VarScale) 
        (*vec_buff)[i] = BaseFloat(sqrt((*vec_buff)[i]));
    }
    
    if (fscanf(fp, "%64s", s2) == 1) 
    {
      throw std::runtime_error(std::string("End of file expected but '") 
          + s2 + "' found in " + typeStr2 + " file " + pFileName);
    }
    
    fclose(fp);
  } // ReadCepsNormFile(...)
  

  //***************************************************************************
  //***************************************************************************
  void
  FeatureRepository::
  HtkFilter(const char* pFilter, const char* pValue, FeatureRepository& rOut)
  {
    std::list<FileListElem>::iterator   it;
    std::string  str;

    rOut.mSwapFeatures    = mSwapFeatures;
    rOut.mStartFrameExt   = mStartFrameExt;
    rOut.mEndFrameExt     = mEndFrameExt;
    rOut.mTargetKind      = mTargetKind;
    rOut.mDerivOrder      = mDerivOrder;
    rOut.mDerivWinLengths = mDerivWinLengths;

    rOut.mpCvgFile        = mpCvgFile;
    rOut.mpCmnPath        = mpCmnPath;
    rOut.mpCmnMask        = mpCmnMask;
    rOut.mpCvnPath        = mpCvnPath;
    rOut.mpCvnMask        = mpCvnMask;

    rOut.mInputQueue.clear();

    // go through all records and check the mask
    for (it=mInputQueue.begin(); it!= mInputQueue.end(); ++it) {
      if (pFilter == NULL
      ||  (ProcessMask(it->Logical(), pFilter, str) && (str == pValue))) {
        rOut.mInputQueue.push_back(*it);
      }
    }

    // set the queue position to the begining
    rOut.mInputQueueIterator = mInputQueue.end(); 

    rOut.mCurrentIndexFileName  = "";
    rOut.mCurrentIndexFileDir   = "";
    rOut.mCurrentIndexFileExt   = "";

    mStream.close();
    mStream.clear();

    rOut.mpLastFileName = NULL;
    rOut.mLastFileName  = "";
    rOut.mpLastCmnFile  = NULL;
    rOut.mpLastCvnFile  = NULL;
    rOut.mpLastCvgFile  = NULL;
    rOut.mpCmn          = NULL;
    rOut.mpCvn          = NULL;
    rOut.mpCvg          = NULL;
    rOut.mpA            = NULL;
    rOut.mpB            = NULL;

  }


  //***************************************************************************
  //***************************************************************************
  void
  FeatureRepository::
  HtkSelection(const char* pFilter, std::list< std::string >& rOut)
  {
    std::map< std::string, bool> aux_map;
    std::map< std::string, bool>::iterator map_it;
    std::list<FileListElem>::iterator   it;
    std::string  str;

    rOut.clear();
    
    if(pFilter != NULL) {
      // go through all records and check the mask
      for (it=mInputQueue.begin(); it!= mInputQueue.end(); ++it) {
        if (ProcessMask(it->Logical(), pFilter, str)) {
          aux_map[str] = true;
        }
      }
    } else {
      aux_map[std::string("default speaker")] = true;
    }

    for (map_it = aux_map.begin(); map_it != aux_map.end(); ++map_it) {
      rOut.push_back(map_it->first);
    }
  }


  //***************************************************************************
  //***************************************************************************
  int     
  FeatureRepository::
  ParmKind2Str(unsigned parmKind, char *pOutString)
  {
    // :KLUDGE: Absolutely no idea what this is...
      if ((parmKind & 0x003F) >= sizeof(mpParmKindNames)/sizeof(mpParmKindNames[0])) 
      return 0;
  
    strcpy(pOutString, mpParmKindNames[parmKind & 0x003F]);
  
    if (parmKind & PARAMKIND_E) strcat(pOutString, "_E");
    if (parmKind & PARAMKIND_N) strcat(pOutString, "_N");
    if (parmKind & PARAMKIND_D) strcat(pOutString, "_D");
    if (parmKind & PARAMKIND_A) strcat(pOutString, "_A");
    if (parmKind & PARAMKIND_C) strcat(pOutString, "_C");
    if (parmKind & PARAMKIND_Z) strcat(pOutString, "_Z");
    if (parmKind & PARAMKIND_K) strcat(pOutString, "_K");
    if (parmKind & PARAMKIND_0) strcat(pOutString, "_0");
    if (parmKind & PARAMKIND_V) strcat(pOutString, "_V");
    if (parmKind & PARAMKIND_T) strcat(pOutString, "_T");
    
    return 1;
  }


  // //***************************************************************************
  // //***************************************************************************
  // void
  // AddFileListToFeatureRepositories(
  //   const char* pFileName, 
  //   const char* pFilter, 
  //   std::queue<FeatureRepository *> &featureRepositoryList)
  // {
  //   IStkStream            l_stream;
  //   std::string           file_name;
  //   Tokenizer             file_list(pFileName, ",");
  //   Tokenizer::iterator   p_file_name;

  //   //:TODO: error if empty featureRepositoryList
  //   
  //   for (p_file_name = file_list.begin(); p_file_name != file_list.end(); ++p_file_name)
  //   {
  //     // get rid of initial and trailing blanks
  //     Trim(*p_file_name);

  //     // open file name
  //     l_stream.open(p_file_name->c_str(), std::ios::in, pFilter);
  //     
  //     if (!l_stream.good()) {
  //       //:TODO:
  //       // Warning or error ... Why warning? -Lukas
  //       throw std::runtime_error(std::string("Cannot not open list file ") +
  //           *p_file_name);
  //     }

  //     // read all lines and parse them
  //     for(;;)
  //     {
  //       l_stream >> file_name;
  //       //:TODO: if(l_stream.badl()) Error()
  //       // Reading after last token set the fail bit
  //       if(l_stream.fail()) 
  //         break;
  //       // we can push_back a std::string as new FileListElem object
  //       // is created using FileListElem(const std::string&) constructor
  //       // and logical and physical names are correctly extracted
  //       featureRepositoryList.front()->mInputQueue.push_back(file_name);
  //       
  //       //cycle in the featureRepositoryList
  //       featureRepositoryList.push(featureRepositoryList.front());
  //       featureRepositoryList.pop();
  //     }
  //     l_stream.close();
  //   }
  // } // AddFileList(const std::string & rFileName)


  //***************************************************************************
  //***************************************************************************
  void
  FeatureRepository::
  Init(
      bool                  swap,
      int                   extLeft,
      int                   extRight,
      int                   targetKind,
      int                   derivOrder,
      int*                  pDerivWinLen,
      const char*           pCmnPath,
      const char*           pCmnMask,
      const char*           pCvnPath,
      const char*           pCvnMask,
      const char*           pCvgFile)
  {
    mSwapFeatures       =   swap;         
    mStartFrameExt      =   extLeft;      
    mEndFrameExt        =   extRight;     
    mTargetKind         =   targetKind;   
    mDerivOrder         =   derivOrder;   
    mDerivWinLengths    =   pDerivWinLen; 
    mpCmnPath           =   pCmnPath;     
    mpCmnMask           =   pCmnMask;     
    mpCvnPath           =   pCvnPath;     
    mpCvnMask           =   pCvnMask;     
    mpCvgFile           =   pCvgFile;    
  } // Init()


  //***************************************************************************
  //***************************************************************************
  void
  FeatureRepository::
  AddFile(const std::string & rFileName)
  {
    mInputQueue.push_back(rFileName);
  } // AddFile(const std::string & rFileName)

  
  //***************************************************************************
  //***************************************************************************
  void
  FeatureRepository::
  AddFileList(const char* pFileName, const char* pFilter)
  {
    IStkStream            l_stream;
    std::string           file_name;
    Tokenizer             file_list(pFileName, ",");
    Tokenizer::iterator   p_file_name;

    for (p_file_name = file_list.begin(); p_file_name != file_list.end(); ++p_file_name)
    {
      // get rid of spaces
      Trim(*p_file_name);

      // open the file
      l_stream.open(p_file_name->c_str(), std::ios::in, pFilter);
      
      if (!l_stream.good())
      {
        //:TODO:
        // Warning or error ... Why warning? -Lukas
        throw std::runtime_error(std::string("Cannot not open list file ") +
            *p_file_name);
      }
      // read all lines and parse them
      for(;;)
      {
        l_stream >> file_name;
        //:TODO: if(l_stream.badl()) Error()
        // Reading after last token set the fail bit
        if(l_stream.fail()) 
	  break;
        // we can push_back a std::string as new FileListElem object
        // is created using FileListElem(const std::string&) constructor
        // and logical and physical names are correctly extracted
        mInputQueue.push_back(file_name);
      }
      l_stream.close();
    }
  } // AddFileList(const std::string & rFileName)

  
  //***************************************************************************
  //***************************************************************************
  void
  FeatureRepository::
  MoveNext()
  {
    assert (mInputQueueIterator != mInputQueue.end());
    mInputQueueIterator++;
  } // ReadFullMatrix(Matrix<BaseFloat>& rMatrix)


  //***************************************************************************
  //***************************************************************************
  bool
  FeatureRepository::
  ReadFullMatrix(Matrix<BaseFloat>& rMatrix)
  {
    // clear the matrix
    rMatrix.Destroy();

    // extract index file name
    if (!mCurrentIndexFileDir.empty())
    {
      char tmp_name[mCurrentIndexFileDir.length() + 
        mCurrentIndexFileExt.length() + 
        mInputQueueIterator->Physical().length()]; 
      
      MakeHtkFileName(tmp_name, mInputQueueIterator->Physical().c_str(), 
          mCurrentIndexFileDir.c_str(), mCurrentIndexFileExt.c_str());
      
      mCurrentIndexFileName = tmp_name;
    }
    else
      mCurrentIndexFileName = "";

    //get the 3-letter suffix
    int pos_last_three_chars = mInputQueueIterator->Physical().size() - 3;
    if (pos_last_three_chars < 0) pos_last_three_chars = 0;
    //read the gzipped ascii features
    if (mInputQueueIterator->Physical().substr(pos_last_three_chars) == ".gz") {
      return ReadGzipAsciiFeatures(*mInputQueueIterator, rMatrix);
    }
     
    // read the matrix and return the result
    return ReadHTKFeatures(*mInputQueueIterator, rMatrix);
  } // ReadFullMatrix(Matrix<BaseFloat>& rMatrix)



  //***************************************************************************
  //***************************************************************************
  bool
  FeatureRepository::
  WriteFeatureMatrix(const Matrix<BaseFloat>& rMatrix, const std::string& filename, int targetKind, int samplePeriod)
  {
    FILE* fp = fopen(filename.c_str(),"w");
    if(NULL == fp) { Error(std::string("Cannot create file:") + filename); return false; }

    WriteHTKFeatures(fp, samplePeriod, targetKind, mSwapFeatures, const_cast<Matrix<BaseFloat>&>(rMatrix));

    fclose(fp);

    return true;
  }


  //***************************************************************************
  //***************************************************************************
  // private:
  int 
  FeatureRepository::
  ReadHTKHeader()
  {
    // TODO 
    // Change this... We should read from StkStream
    FILE* fp = mStream.fp();
    
    if (!fread(&mHeader.mNSamples,     sizeof(INT_32),  1, fp)) return -1;
    if (!fread(&mHeader.mSamplePeriod, sizeof(INT_32),  1, fp)) return -1;
    if (!fread(&mHeader.mSampleSize,   sizeof(INT_16),  1, fp)) return -1;
    if (!fread(&mHeader.mSampleKind,   sizeof(UINT_16), 1, fp)) return -1;

    if (mSwapFeatures) 
    {
      swap4(mHeader.mNSamples);
      swap4(mHeader.mSamplePeriod);
      swap2(mHeader.mSampleSize);
      swap2(mHeader.mSampleKind);
    }
  
    if (mHeader.mSamplePeriod < 0 
    ||  mHeader.mSamplePeriod > 100000 
    ||  mHeader.mNSamples     < 0 
    ||  mHeader.mSampleSize   < 0) 
    {
      return -1;
    }
  
    return 0;
  }
  
  
  //***************************************************************************
  //***************************************************************************
  // private:
  int 
  FeatureRepository::
  ReadHTKFeature(
      BaseFloat*    pIn, 
      size_t    feaLen, 
      bool      decompress, 
      BaseFloat*    pScale, 
      BaseFloat*    pBias)
  {
    FILE*  fp = mStream.fp();
    
    size_t i;
    
    if (decompress) 
    {
      INT_16 s;
  //    BaseFloat pScale = (xmax - xmin) / (2*32767);
  //    BaseFloat pBias  = (xmax + xmin) / 2;
  
      for (i = 0; i < feaLen; i++) 
      {
        if (fread(&s, sizeof(INT_16), 1, fp) != 1) 
          return -1;
        
        if (mSwapFeatures) swap2(s);
        pIn[i] = ((BaseFloat)s + pBias[i]) / pScale[i];
      }
      
      return 0;
    }
  
#if !DOUBLEPRECISION
    if (fread(pIn, sizeof(FLOAT_32), feaLen, fp) != feaLen) 
      return -1;
    
    if (mSwapFeatures) 
      for (i = 0; i < feaLen; i++) 
        swap4(pIn[i]);
#else
    float f;
  
    for (i = 0; i < feaLen; i++) 
    {
      if (fread(&f, sizeof(FLOAT_32), 1, fp) != 1)
        return -1;
      
      if (mSwapFeatures) 
        swap4(f);
        
      pIn[i] = f;
    }
#endif
    return 0;
  }  // int ReadHTKFeature
  
  

  //***************************************************************************
  //***************************************************************************
/*  bool 
  FeatureRepository::
  ReadHTKFeatures(const std::string& rFileName, Matrix<BaseFloat>& rFeatureMatrix)
  {
    std::string           file_name(rFileName);
    std::string           cmn_file_name;
    std::string           cvn_file_name;  
    
    int                   ext_left  = mStartFrameExt;
    int                   ext_right = mEndFrameExt;
    int                   from_frame;
    int                   to_frame;
    int                   tot_frames;
    int                   trg_vec_size;
    int                   src_vec_size;
    int                   src_deriv_order;
    int                   lo_src_tgz_deriv_order;
    int                   i;
    int                   j;
    int                   k;
    int                   e;
    int                   coefs;
    int                   trg_E;
    int                   trg_0;
    int                   trg_N;
    int                   src_E;
    int                   src_0;
    int                   src_N;
    int                   comp;
    int                   coef_size;
    char*                 chptr;
  
    
   
    // read frame range definition if any ( physical_file.fea[s,e] )
    if ((chptr = strrchr(file_name.c_str(), '[')) == NULL ||
        ((i=0), sscanf(chptr, "[%d,%d]%n", &from_frame, &to_frame, &i), 
         chptr[i] != '\0')) 
    {
      chptr = NULL;
    }
    
    if (chptr != NULL)                                
      *chptr = '\0';
  
  // Experimental changes...
  // if ((strcmp(file_name.c_str(), "-"))
  // &&  (mpLastFileName != NULL) 
  // &&  (!strcmp(mpLastFileName, file_name.c_str()))) 
  //   {
  //     mHeader = mLastHeader;
  //   } 
  //   else 
  //   {
  //     if (mpLastFileName) 
  //     {
  //       //if (mpFp != stdin) 
  //       //  fclose(mpFp);
  //       mStream.close();
  //       
  //       free(mpLastFileName);
  //       mpLastFileName = NULL;
  //     }

    if ((file_name != "-" )
    &&  (!mLastFileName.empty()) 
    &&  (mLastFileName == file_name)) 
    {
      mHeader = mLastHeader;
    } 
    else 
    {
      if (!mLastFileName.empty()) 
      {
        mStream.close();
        mLastFileName = "";
      }
      
      
      // open the feature file
      mStream.open(file_name.c_str(), ios::binary);
      if (!mStream.good())
      {
        Error("Cannot open feature file: '%s'", file_name.c_str());
      }
      
      
      if (ReadHTKHeader()) 
        Error("Invalid HTK header in feature file: '%s'", file_name.c_str());
      
      if (mHeader.mSampleKind & PARAMKIND_C) 
      {
        // File is in compressed form, scale and pBias vectors
        // are appended after HTK header.
  
        int coefs = mHeader.mSampleSize/sizeof(INT_16);
        mpA = (BaseFloat*) realloc(mpA, coefs * sizeof(BaseFloat));
        mpB = (BaseFloat*) realloc(mpB, coefs * sizeof(BaseFloat));
        if (mpA == NULL || mpB == NULL) Error("Insufficient memory");
  
        e  = ReadHTKFeature(mpA, coefs, 0, 0, 0);
        e |= ReadHTKFeature(mpB, coefs, 0, 0, 0);
        
        if (e) 
          Error("Cannot read feature file: '%s'", file_name.c_str());
        
        mHeader.mNSamples -= 2 * sizeof(FLOAT_32) / sizeof(INT_16);
      }
      
      // remember current settings
      mLastFileName = file_name;
      mLastHeader   = mHeader;
    }
    
    if (chptr != NULL) 
      *chptr = '[';
  
    if (chptr == NULL) 
    { // Range [s,e] was not specified
      from_frame = 0;
      to_frame   = mHeader.mNSamples-1;
    }
    
    src_deriv_order = PARAMKIND_T & mHeader.mSampleKind ? 3 :
                      PARAMKIND_A & mHeader.mSampleKind ? 2 :
                      PARAMKIND_D & mHeader.mSampleKind ? 1 : 0;
    src_E =  (PARAMKIND_E & mHeader.mSampleKind) != 0;
    src_0 =  (PARAMKIND_0 & mHeader.mSampleKind) != 0;
    src_N = ((PARAMKIND_N & mHeader.mSampleKind) != 0) * (src_E + src_0);
    comp =    PARAMKIND_C & mHeader.mSampleKind;
    
    mHeader.mSampleKind &= ~PARAMKIND_C;
  
    if (mTargetKind == PARAMKIND_ANON) 
    {
      mTargetKind = mHeader.mSampleKind;
    } 
    else if ((mTargetKind & 077) == PARAMKIND_ANON) 
    {
      mTargetKind &= ~077;
      mTargetKind |= mHeader.mSampleKind & 077;
    }
    
    trg_E = (PARAMKIND_E & mTargetKind) != 0;
    trg_0 = (PARAMKIND_0 & mTargetKind) != 0;
    trg_N =((PARAMKIND_N & mTargetKind) != 0) * (trg_E + trg_0);
  
    coef_size     = comp ? sizeof(INT_16) : sizeof(FLOAT_32);
    coefs         = (mHeader.mSampleSize/coef_size + src_N) / 
                    (src_deriv_order+1) - src_E - src_0;
    src_vec_size  = (coefs + src_E + src_0) * (src_deriv_order+1) - src_N;
  
    //Is coefs dividable by 1 + number of derivatives specified in header
    if (src_vec_size * coef_size != mHeader.mSampleSize) 
    {
      Error("Invalid HTK header in feature file: '%s'. "
            "mSampleSize do not match with parmKind", file_name.c_str());
    }
    
    if (mDerivOrder < 0) 
      mDerivOrder = src_deriv_order;
  
  
    if ((!src_E && trg_E) || (!src_0 && trg_0) || (src_N && !trg_N) ||
        (trg_N && !trg_E && !trg_0) || (trg_N && !mDerivOrder) ||
        (src_N && !src_deriv_order && mDerivOrder) ||
        ((mHeader.mSampleKind & 077) != (mTargetKind & 077) &&
         (mHeader.mSampleKind & 077) != PARAMKIND_ANON)) 
    {
      char srcParmKind[64];
      char trgParmKind[64];
      
      ParmKind2Str(mHeader.mSampleKind, srcParmKind);
      ParmKind2Str(mTargetKind,       trgParmKind);
      Error("Cannot convert %s to %s", srcParmKind, trgParmKind);
    }
  
    lo_src_tgz_deriv_order = LOWER_OF(src_deriv_order, mDerivOrder);
    trg_vec_size  = (coefs + trg_E + trg_0) * (mDerivOrder+1) - trg_N;
    
    i =  LOWER_OF(from_frame, mStartFrameExt);
    from_frame  -= i;
    ext_left     -= i;
  
    i =  LOWER_OF(mHeader.mNSamples-to_frame-1, mEndFrameExt);
    to_frame    += i;
    ext_right    -= i;
  
    if (from_frame > to_frame || from_frame >= mHeader.mNSamples || to_frame< 0)
      Error("Invalid frame range for feature file: '%s'", file_name.c_str());
    
    tot_frames = to_frame - from_frame + 1 + ext_left + ext_right;
    
    // initialize matrix 
    rFeatureMatrix.Init(tot_frames, trg_vec_size);
    
    // fill the matrix with features
    for (i = 0; i <= to_frame - from_frame; i++) 
    {
      BaseFloat* A      = mpA;
      BaseFloat* B      = mpB;
      BaseFloat* mxPtr  = rFeatureMatrix[i+ext_left];
      
      // seek to the desired position
      fseek(mStream.fp(), 
          sizeof(HtkHeader) + (comp ? src_vec_size * 2 * sizeof(FLOAT_32) : 0)
          + (from_frame + i) * src_vec_size * coef_size, 
          SEEK_SET);
  
      e = ReadHTKFeature(mxPtr, coefs, comp, A, B);
      
      mxPtr += coefs; 
      A     += coefs; 
      B     += coefs;
        
      if (src_0 && !src_N) e |= ReadHTKFeature(mxPtr, 1, comp, A++, B++);
      if (trg_0 && !trg_N) mxPtr++;
      if (src_E && !src_N) e |= ReadHTKFeature(mxPtr, 1, comp, A++, B++);
      if (trg_E && !trg_N) mxPtr++;
  
      for (j = 0; j < lo_src_tgz_deriv_order; j++) 
      {
        e |= ReadHTKFeature(mxPtr, coefs, comp, A, B);
        mxPtr += coefs; 
        A     += coefs; 
        B     += coefs;
        
        if (src_0) e |= ReadHTKFeature(mxPtr, 1, comp, A++, B++);
        if (trg_0) mxPtr++;
        if (src_E) e |= ReadHTKFeature(mxPtr, 1, comp, A++, B++);
        if (trg_E) mxPtr++;
      }
      
      if (e) 
        Error("Cannot read feature file: '%s' frame %d/%d", file_name.c_str(),
            i, to_frame - from_frame + 1);
    }
  
    // From now, coefs includes also trg_0 + trg_E !
    coefs += trg_0 + trg_E; 
    
    // If extension of the matrix to the left or to the right is required,
    // perform it here
    for (i = 0; i < ext_left; i++) 
    {
      memcpy(rFeatureMatrix[i],
             rFeatureMatrix[ext_left],
             (coefs * (1+lo_src_tgz_deriv_order) - trg_N) * sizeof(BaseFloat));
    }
    
    for (i = tot_frames - ext_right; i < tot_frames; i++) 
    {
      memcpy(rFeatureMatrix[i],
             rFeatureMatrix[tot_frames - ext_right - 1],
             (coefs * (1+lo_src_tgz_deriv_order) - trg_N) * sizeof(BaseFloat));
    }

    // Sentence cepstral mean normalization
    if( (mpCmnPath == NULL)
    && !(PARAMKIND_Z & mHeader.mSampleKind) 
    &&  (PARAMKIND_Z & mTargetKind)) 
    {
      // for each coefficient
      for(j=0; j < coefs; j++) 
      {          
        BaseFloat norm = 0.0;
        for(i=0; i < tot_frames; i++)      // for each frame
        {
          norm += rFeatureMatrix[i][j - trg_N];
          //norm += fea_mx[i*trg_vec_size - trg_N + j];
        }
        
        norm /= tot_frames;
  
        for(i=0; i < tot_frames; i++)      // for each frame
          rFeatureMatrix[i][j - trg_N] -= norm;
          //fea_mx[i*trg_vec_size - trg_N + j] -= norm;
      }
    }
    
    // Compute missing derivatives
    for (; src_deriv_order < mDerivOrder; src_deriv_order++) 
    { 
      int winLen = mDerivWinLengths[src_deriv_order];
      BaseFloat norm = 0.0;
      
      for (k = 1; k <= winLen; k++) 
      {
        norm += 2 * k * k;
      }
      
      // for each frame
      for (i=0; i < tot_frames; i++) 
      {        
        // for each coefficient
        for (j=0; j < coefs; j++) 
        {          
          //BaseFloat* src = fea_mx + i*trg_vec_size + src_deriv_order*coefs - trg_N + j;
          BaseFloat* src = &rFeatureMatrix[i][src_deriv_order*coefs - trg_N + j];
          
          *(src + coefs) = 0.0;
          
          if (i < winLen || i >= tot_frames-winLen) 
          { // boundaries need special treatment
            for (k = 1; k <= winLen; k++) 
            {  
              *(src+coefs) += k*(src[ LOWER_OF(tot_frames-1-i,k)*rFeatureMatrix.Stride()]
                                -src[-LOWER_OF(i,             k)*rFeatureMatrix.Stride()]);
            }
          } 
          else 
          { // otherwise use more efficient code
            for (k = 1; k <= winLen; k++) 
            {  
              *(src+coefs) += k*(src[ k * rFeatureMatrix.Stride()]
                                -src[-k * rFeatureMatrix.Stride()]);
            }
          }
          *(src + coefs) /= norm;
        }
      }
    }
    
    mHeader.mNSamples    = tot_frames;
    mHeader.mSampleSize  = trg_vec_size * sizeof(FLOAT_32);
    mHeader.mSampleKind  = mTargetKind & ~(PARAMKIND_D | PARAMKIND_A | PARAMKIND_T);
  

    ////////////////////////////////////////////////////////////////////////////
    /////////////// Cepstral mean and variance normalization ///////////////////
    ////////////////////////////////////////////////////////////////////////////
    //.........................................................................
    if (mpCmnPath != NULL
    &&  mpCmnMask != NULL) 
    {
      // retrieve file name
      ProcessMask(file_name, mpCmnMask, cmn_file_name);
      // add the path correctly
      cmn_file_name.insert(0, "/");
      cmn_file_name.insert(0, mpCmnPath);

      // read the file
      ReadCepsNormFile(cmn_file_name.c_str(), &mpLastCmnFile, &mpCmn,
          mHeader.mSampleKind & ~PARAMKIND_Z, CNF_Mean, coefs);
                      
      // recompute feature values
      for (i=0; i < tot_frames; i++) 
      {
        for (j=trg_N; j < coefs; j++) 
        {
          rFeatureMatrix[i][j - trg_N] -= mpCmn[j];
        }
      }
    }
  
    mHeader.mSampleKind |= mDerivOrder==3 ? PARAMKIND_D | PARAMKIND_A | PARAMKIND_T :
                           mDerivOrder==2 ? PARAMKIND_D | PARAMKIND_A :
                           mDerivOrder==1 ? PARAMKIND_D : 0;
  
    //.........................................................................
    if (mpCvnPath != NULL
    &&  mpCvnMask != NULL) 
    {
      // retrieve file name
      ProcessMask(file_name, mpCvnMask, cvn_file_name);
      // add the path correctly
      cvn_file_name.insert(0, "/");
      cvn_file_name.insert(0, mpCvnPath);

      // read the file
      ReadCepsNormFile(cvn_file_name.c_str(), &mpLastCvnFile, &mpCvn,
          mHeader.mSampleKind, CNF_Variance, trg_vec_size);
                      
      // recompute feature values
      for (i=0; i < tot_frames; i++) 
      {
        for (j=trg_N; j < trg_vec_size; j++) 
        {
          rFeatureMatrix[i][j - trg_N] *= mpCvn[j];
        }
      }
    }
    
    //.........................................................................
    // process the global covariance file
    if (mpCvgFile != NULL) 
    {
      ReadCepsNormFile(mpCvgFile, &mpLastCvgFile, &mpCvg,
                      -1, CNF_VarScale, trg_vec_size);
                      
      // recompute feature values
      for (i=0; i < tot_frames; i++) 
      {
        for (j=trg_N; j < trg_vec_size; j++) 
        {
          rFeatureMatrix[i][j - trg_N] *= mpCvg[j];
        }
      }
    }
    
    return true;
  }
*/

  //***************************************************************************
  //***************************************************************************




  
  //***************************************************************************
  //***************************************************************************
  bool 
  FeatureRepository::
  ReadHTKFeatures(const FileListElem&    rFileNameRecord, 
                        Matrix<BaseFloat>&        rFeatureMatrix)
  {
    std::string           file_name(rFileNameRecord.Physical());
    std::string           cmn_file_name;
    std::string           cvn_file_name;  
    
    int                   ext_left  = mStartFrameExt;
    int                   ext_right = mEndFrameExt;
    int                   from_frame;
    int                   to_frame;
    int                   tot_frames;
    int                   trg_vec_size;
    int                   src_vec_size;
    int                   src_deriv_order;
    int                   lo_src_tgz_deriv_order;
    int                   i;
    int                   j;
    int                   k;
    int                   e;
    int                   coefs;
    int                   trg_E;
    int                   trg_0;
    int                   trg_N;
    int                   src_E;
    int                   src_0;
    int                   src_N;
    int                   comp;
    int                   coef_size;
    char*                 chptr;
  

  TIMER_START(mTim);
   
    // read frame range definition if any ( physical_file.fea[s,e] )
    if ((chptr = strrchr((char*)file_name.c_str(), '[')) == NULL ||
        ((i=0), sscanf(chptr, "[%d,%d]%n", &from_frame, &to_frame, &i), 
         chptr[i] != '\0')) 
    {
      chptr = NULL;
    }
    
    if (chptr != NULL)                                
      *chptr = '\0';
  

    if ((file_name != "-" )
    &&  (!mLastFileName.empty()) 
    &&  (mLastFileName == file_name)) 
    {
      mHeader = mLastHeader;
    } 
    else 
    {
      if (!mLastFileName.empty()) 
      {
        mStream.close();
        mLastFileName = "";
      }
      
      
      // open the feature file
      mStream.open(file_name.c_str(), std::ios::binary);
      if (!mStream.good())
      {
        throw std::runtime_error(std::string("Cannot open feature file: '") 
            + file_name.c_str() + "'");
      }
      
      
      if (ReadHTKHeader())  {
        throw std::runtime_error(std::string("Invalid HTK header in feature file: '") 
            + file_name.c_str() + "'");
      }
      
      if (mHeader.mSampleKind & PARAMKIND_C) 
      {
        // File is in compressed form, scale and pBias vectors
        // are appended after HTK header.
	    coefs = mHeader.mSampleSize/sizeof(INT_16);

        mpA = (BaseFloat*) realloc(mpA, coefs * sizeof(BaseFloat));
        mpB = (BaseFloat*) realloc(mpB, coefs * sizeof(BaseFloat));

        if (mpA == NULL || mpB == NULL) {
          throw std::runtime_error("Insufficient memory");
        }
  
        e  = ReadHTKFeature(mpA, coefs, 0, 0, 0);
        e |= ReadHTKFeature(mpB, coefs, 0, 0, 0);
        
        if (e) {
          throw std::runtime_error(std::string("Cannot read feature file: '") 
              + file_name.c_str() + "'");
        }
        
        mHeader.mNSamples -= 2 * sizeof(FLOAT_32) / sizeof(INT_16);
      }
      
      // remember current settings
      mLastFileName = file_name;
      mLastHeader   = mHeader;
    }
    
    if (chptr != NULL) {
      *chptr = '[';
    }
  
    if (chptr == NULL) { 
      // Range [s,e] was not specified
      from_frame = 0;
      to_frame   = mHeader.mNSamples-1;
    }
    
    src_deriv_order = PARAMKIND_T & mHeader.mSampleKind ? 3 :
                      PARAMKIND_A & mHeader.mSampleKind ? 2 :
                      PARAMKIND_D & mHeader.mSampleKind ? 1 : 0;
    src_E =  (PARAMKIND_E & mHeader.mSampleKind) != 0;
    src_0 =  (PARAMKIND_0 & mHeader.mSampleKind) != 0;
    src_N = ((PARAMKIND_N & mHeader.mSampleKind) != 0) * (src_E + src_0);
    comp =    PARAMKIND_C & mHeader.mSampleKind;
    
    mHeader.mSampleKind &= ~PARAMKIND_C;
  
    if (mTargetKind == PARAMKIND_ANON) 
    {
      mTargetKind = mHeader.mSampleKind;
    } 
    else if ((mTargetKind & 077) == PARAMKIND_ANON) 
    {
      mTargetKind &= ~077;
      mTargetKind |= mHeader.mSampleKind & 077;
    }
    
    trg_E = (PARAMKIND_E & mTargetKind) != 0;
    trg_0 = (PARAMKIND_0 & mTargetKind) != 0;
    trg_N =((PARAMKIND_N & mTargetKind) != 0) * (trg_E + trg_0);
  
    coef_size     = comp ? sizeof(INT_16) : sizeof(FLOAT_32);
    coefs         = (mHeader.mSampleSize/coef_size + src_N) / 
                    (src_deriv_order+1) - src_E - src_0;
    src_vec_size  = (coefs + src_E + src_0) * (src_deriv_order+1) - src_N;
  
    //Is coefs dividable by 1 + number of derivatives specified in header
    if (src_vec_size * coef_size != mHeader.mSampleSize) 
    {
      throw std::runtime_error(std::string("Invalid HTK header in feature file: '") 
            + file_name + "' mSampleSize do not match with parmKind");
    }
    
    if (mDerivOrder < 0) 
      mDerivOrder = src_deriv_order;
  
  
    if ((!src_E && trg_E) || (!src_0 && trg_0) || (src_N && !trg_N) ||
        (trg_N && !trg_E && !trg_0) || (trg_N && !mDerivOrder) ||
        (src_N && !src_deriv_order && mDerivOrder) ||
        ((mHeader.mSampleKind & 077) != (mTargetKind & 077) &&
         (mHeader.mSampleKind & 077) != PARAMKIND_ANON)) 
    {
      char srcParmKind[64];
      char trgParmKind[64];
      memset(srcParmKind,0,64);
      memset(trgParmKind,0,64);
      
      ParmKind2Str(mHeader.mSampleKind, srcParmKind);
      ParmKind2Str(mTargetKind,       trgParmKind);
      throw std::runtime_error(std::string("Cannot convert ") + srcParmKind 
          + " to " + trgParmKind);
    }
  
    lo_src_tgz_deriv_order = std::min(src_deriv_order, mDerivOrder);
    trg_vec_size  = (coefs + trg_E + trg_0) * (mDerivOrder+1) - trg_N;
    
    i =  std::min(from_frame, mStartFrameExt);
    from_frame  -= i;
    ext_left     -= i;
  
    i =  std::min(mHeader.mNSamples-to_frame-1, mEndFrameExt);
    to_frame    += i;
    ext_right    -= i;
  
    if (from_frame > to_frame || from_frame >= mHeader.mNSamples || to_frame< 0)
      throw std::runtime_error(std::string("Invalid frame range for feature file: '")
            + file_name.c_str() + "'");
    
    tot_frames = to_frame - from_frame + 1 + ext_left + ext_right;
   
    
   TIMER_END(mTim,mTimeOpen);


    // initialize matrix 
    rFeatureMatrix.Init(tot_frames, trg_vec_size, false);
    
    // fill the matrix with features
    for (i = 0; i <= to_frame - from_frame; i++) 
    {
      BaseFloat* A      = mpA;
      BaseFloat* B      = mpB;
      BaseFloat* mxPtr  = rFeatureMatrix.pRowData(i+ext_left);

    TIMER_START(mTim);      
      // seek to the desired position
      fseek(mStream.fp(), 
          sizeof(HtkHeader) + (comp ? src_vec_size * 2 * sizeof(FLOAT_32) : 0)
          + (from_frame + i) * src_vec_size * coef_size, 
          SEEK_SET);
    TIMER_END(mTim,mTimeSeek);
 
    TIMER_START(mTim);
      // read 
      e = ReadHTKFeature(mxPtr, coefs, comp, A, B);
    TIMER_END(mTim,mTimeRead);
      
      mxPtr += coefs; 
      A     += coefs; 
      B     += coefs;
        
      if (src_0 && !src_N) e |= ReadHTKFeature(mxPtr, 1, comp, A++, B++);
      if (trg_0 && !trg_N) mxPtr++;
      if (src_E && !src_N) e |= ReadHTKFeature(mxPtr, 1, comp, A++, B++);
      if (trg_E && !trg_N) mxPtr++;
  
      for (j = 0; j < lo_src_tgz_deriv_order; j++) 
      {
        e |= ReadHTKFeature(mxPtr, coefs, comp, A, B);
        mxPtr += coefs; 
        A     += coefs; 
        B     += coefs;
        
        if (src_0) e |= ReadHTKFeature(mxPtr, 1, comp, A++, B++);
        if (trg_0) mxPtr++;
        if (src_E) e |= ReadHTKFeature(mxPtr, 1, comp, A++, B++);
        if (trg_E) mxPtr++;
      }

      if (e) {
        std::cout << mHeader.mNSamples << "\n";
        std::cout << 2 * sizeof(FLOAT_32) / sizeof(INT_16) << "\n";
        std::cout << "from" << from_frame << "to" << to_frame << "i" << i << "\n";

        std::ostringstream s;
        s << i << "/" << to_frame - from_frame + 1, s.str();
        throw std::runtime_error(std::string("Cannot read feature file: '")
              + file_name + "' frame " + s.str());
      }
    }
  
    // From now, coefs includes also trg_0 + trg_E !
    coefs += trg_0 + trg_E; 
    
    // If extension of the matrix to the left or to the right is required,
    // perform it here
    for (i = 0; i < ext_left; i++) 
    {
      memcpy(rFeatureMatrix.pRowData(i),
             rFeatureMatrix.pRowData(ext_left),
             (coefs * (1+lo_src_tgz_deriv_order) - trg_N) * sizeof(BaseFloat));
    }
    
    for (i = tot_frames - ext_right; i < tot_frames; i++) 
    {
      memcpy(rFeatureMatrix.pRowData(i),
             rFeatureMatrix.pRowData(tot_frames - ext_right - 1),
             (coefs * (1+lo_src_tgz_deriv_order) - trg_N) * sizeof(BaseFloat));
    }

    // Sentence cepstral mean normalization
    if( (mpCmnPath == NULL)
    && !(PARAMKIND_Z & mHeader.mSampleKind) 
    &&  (PARAMKIND_Z & mTargetKind)) 
    {
      // for each coefficient
      for(j=0; j < coefs; j++) 
      {          
        BaseFloat norm = 0.0;
        for(i=0; i < tot_frames; i++)      // for each frame
        {
          norm += rFeatureMatrix[i][j - trg_N];
          //norm += fea_mx[i*trg_vec_size - trg_N + j];
        }
        
        norm /= tot_frames;
  
        for(i=0; i < tot_frames; i++)      // for each frame
          rFeatureMatrix[i][j - trg_N] -= norm;
          //fea_mx[i*trg_vec_size - trg_N + j] -= norm;
      }
    }
    
    // Compute missing derivatives
    for (; src_deriv_order < mDerivOrder; src_deriv_order++) 
    { 
      int winLen = mDerivWinLengths[src_deriv_order];
      BaseFloat norm = 0.0;
      
      for (k = 1; k <= winLen; k++) 
      {
        norm += 2 * k * k;
      }
      
      // for each frame
      for (i=0; i < tot_frames; i++) 
      {        
        // for each coefficient
        for (j=0; j < coefs; j++) 
        {          
          //BaseFloat* src = fea_mx + i*trg_vec_size + src_deriv_order*coefs - trg_N + j;
          BaseFloat* src = &rFeatureMatrix[i][src_deriv_order*coefs - trg_N + j];
          
          *(src + coefs) = 0.0;
          
          if (i < winLen || i >= tot_frames-winLen) 
          { // boundaries need special treatment
            for (k = 1; k <= winLen; k++) 
            {  
              *(src+coefs) += k*(src[ std::min(tot_frames-1-i,k)*rFeatureMatrix.Stride()]
                                -src[-std::min(i,             k)*rFeatureMatrix.Stride()]);
            }
          } 
          else 
          { // otherwise use more efficient code
            for (k = 1; k <= winLen; k++) 
            {  
              *(src+coefs) += k*(src[ k * rFeatureMatrix.Stride()]
                                -src[-k * rFeatureMatrix.Stride()]);
            }
          }
          *(src + coefs) /= norm;
        }
      }
    }
    
    mHeader.mNSamples    = tot_frames;
    mHeader.mSampleSize  = trg_vec_size * sizeof(FLOAT_32);
    mHeader.mSampleKind  = mTargetKind & ~(PARAMKIND_D | PARAMKIND_A | PARAMKIND_T);
  

   TIMER_START(mTim);
    ////////////////////////////////////////////////////////////////////////////
    /////////////// Cepstral mean and variance normalization ///////////////////
    ////////////////////////////////////////////////////////////////////////////
    //.........................................................................
    if (mpCmnPath != NULL
    &&  mpCmnMask != NULL) 
    {
      // retrieve file name
      ProcessMask(rFileNameRecord.Logical(), mpCmnMask, cmn_file_name);
      // add the path correctly

      if(cmn_file_name == "") {
        throw std::runtime_error("CMN Matching failed");
      }

      cmn_file_name.insert(0, "/");
      cmn_file_name.insert(0, mpCmnPath);

      // read the file
      ReadCepsNormFile(cmn_file_name.c_str(), &mpLastCmnFile, &mpCmn,
          mHeader.mSampleKind & ~PARAMKIND_Z, CNF_Mean, coefs);
                      
      // recompute feature values
      for (i=0; i < tot_frames; i++) 
      {
        for (j=trg_N; j < coefs; j++) 
        {
          rFeatureMatrix[i][j - trg_N] -= mpCmn[j];
        }
      }
    }
  
    mHeader.mSampleKind |= mDerivOrder==3 ? PARAMKIND_D | PARAMKIND_A | PARAMKIND_T :
                           mDerivOrder==2 ? PARAMKIND_D | PARAMKIND_A :
                           mDerivOrder==1 ? PARAMKIND_D : 0;
  
    //.........................................................................
    if (mpCvnPath != NULL
    &&  mpCvnMask != NULL) 
    {
      // retrieve file name
      ProcessMask(rFileNameRecord.Logical(), mpCvnMask, cvn_file_name);
      // add the path correctly
      cvn_file_name.insert(0, "/");
      cvn_file_name.insert(0, mpCvnPath);

      // read the file
      ReadCepsNormFile(cvn_file_name.c_str(), &mpLastCvnFile, &mpCvn,
          mHeader.mSampleKind, CNF_Variance, trg_vec_size);
                      
      // recompute feature values
      for (i=0; i < tot_frames; i++) 
      {
        for (j=trg_N; j < trg_vec_size; j++) 
        {
          rFeatureMatrix[i][j - trg_N] *= mpCvn[j];
        }
      }
    }
    
    //.........................................................................
    // process the global covariance file
    if (mpCvgFile != NULL) 
    {
      ReadCepsNormFile(mpCvgFile, &mpLastCvgFile, &mpCvg,
                      -1, CNF_VarScale, trg_vec_size);
                      
      // recompute feature values
      for (i=0; i < tot_frames; i++) 
      {
        for (j=trg_N; j < trg_vec_size; j++) 
        {
          rFeatureMatrix[i][j - trg_N] *= mpCvg[j];
        }
      }
    }

  TIMER_END(mTim,mTimeNormalize);
    
    return true;
  }


  //***************************************************************************
  //***************************************************************************
  int 
  FeatureRepository::
  ReadParmKind(const char *str, bool checkBrackets)
  {
    unsigned int  i;
    int           parmKind =0;
    int           slen     = strlen(str);
  
    if (checkBrackets) 
    {
      if (str[0] != '<' || str[slen-1] != '>')  return -1;
      str++; slen -= 2;
    }
    
    for (; slen >= 0 && str[slen-2] == '_'; slen -= 2) 
    {
      parmKind |= str[slen-1] == 'E' ? PARAMKIND_E :
                  str[slen-1] == 'N' ? PARAMKIND_N :
                  str[slen-1] == 'D' ? PARAMKIND_D :
                  str[slen-1] == 'A' ? PARAMKIND_A :
                  str[slen-1] == 'C' ? PARAMKIND_C :
                  str[slen-1] == 'Z' ? PARAMKIND_Z :
                  str[slen-1] == 'K' ? PARAMKIND_K :
                  str[slen-1] == '0' ? PARAMKIND_0 :
                  str[slen-1] == 'V' ? PARAMKIND_V :
                  str[slen-1] == 'T' ? PARAMKIND_T : -1;
  
      if (parmKind == -1) return -1;
    }
    
    for (i = 0; i < sizeof(mpParmKindNames) / sizeof(char*); i++) 
    {
      if (!strncmp(str, mpParmKindNames[i], slen))
        return parmKind | i;
    }
    return -1;
  }




  //***************************************************************************
  //***************************************************************************
  int
  FeatureRepository:: 
  WriteHTKHeader (FILE * pOutFp, HtkHeader header, bool swap)
  {
    int cc;
  
    if (swap) {
      swap4(header.mNSamples);
      swap4(header.mSamplePeriod);
      swap2(header.mSampleSize);
      swap2(header.mSampleKind);
    }
  
    fseek (pOutFp, 0L, SEEK_SET);
    cc = fwrite(&header, sizeof(HtkHeader), 1, pOutFp);
  
    if (swap) {
      swap4(header.mNSamples);
      swap4(header.mSamplePeriod);
      swap2(header.mSampleSize);
      swap2(header.mSampleKind);
    }
  
    return cc == 1 ? 0 : -1;
  }
  
  
  //***************************************************************************
  //***************************************************************************
  int 
  FeatureRepository::
  WriteHTKFeature(
    FILE * pOutFp,
    FLOAT * pOut,
    size_t feaLen,
    bool swap,
    bool compress,
    FLOAT* pScale, 
    FLOAT* pBias)
  {
    size_t    i;
    size_t    cc = 0;


    if (compress) 
    {
      INT_16 s;
        
      for (i = 0; i < feaLen; i++) 
      {
	s = pOut[i] * pScale[i] - pBias[i];
        if (swap) 
	  swap2(s);
	cc += fwrite(&s, sizeof(INT_16), 1, pOutFp);
      }
      
    } else {
  #if !DOUBLEPRECISION
      if (swap) 
        for (i = 0; i < feaLen; i++) 
          swap4(pOut[i]);
    
        cc = fwrite(pOut, sizeof(FLOAT_32), feaLen, pOutFp);
    
      if (swap) 
        for (i = 0; i < feaLen; i++) 
          swap4(pOut[i]);
  #else
      FLOAT_32 f;
  
      for (i = 0; i < feaLen; i++) 
      {
        f = pOut[i];
        if (swap) 
          swap4(f);
        cc += fwrite(&f, sizeof(FLOAT_32), 1, pOutFp);
      }
  #endif
    }
    return cc == feaLen ? 0 : -1;
  }

  //***************************************************************************
  //***************************************************************************
  int 
  FeatureRepository::
  WriteHTKFeatures(
    FILE *  pOutFp,
    FLOAT * pOut,
    int     nCoeffs,
    int     nSamples,
    int     samplePeriod,
    int     targetKind,  
    bool    swap) 
  {
    HtkHeader header;
    int i, j;
    FLOAT *pScale = NULL;
    FLOAT *pBias = NULL;
    
    header.mNSamples = nSamples  + ((targetKind & PARAMKIND_C) ? 2 * sizeof(FLOAT_32) / sizeof(INT_16) : 0);
    header.mSamplePeriod = samplePeriod;
    header.mSampleSize = nCoeffs * ((targetKind & PARAMKIND_C) ?    sizeof(INT_16)   : sizeof(FLOAT_32));;
    header.mSampleKind = targetKind;
    
    WriteHTKHeader (pOutFp, header, swap);

    if(targetKind & PARAMKIND_C) {
      pScale = (FLOAT*) malloc(nCoeffs * sizeof(FLOAT));
      pBias = (FLOAT*)  malloc(nCoeffs * sizeof(FLOAT));
      if (pScale == NULL || pBias == NULL) Error("Insufficient memory");
      
      for(i = 0; i < nCoeffs; i++) {
        float xmin, xmax;
	xmin = xmax = pOut[i];
	for(j = 1; j < nSamples; j++) {
	  if(pOut[j*nCoeffs+i] > xmax) xmax = pOut[j*nCoeffs+i];
	  if(pOut[j*nCoeffs+i] < xmin) xmin = pOut[j*nCoeffs+i];
	}
	pScale[i] = (2*32767) / (xmax - xmin);
        pBias[i]  = pScale[i] * (xmax + xmin) / 2;
	
	
      }
      if (WriteHTKFeature(pOutFp, pScale, nCoeffs, swap, false, 0, 0)
      ||  WriteHTKFeature(pOutFp, pBias,  nCoeffs, swap, false, 0, 0)) {
        return -1;
      }
    }
    for(j = 0; j < nSamples; j++) {
      if (WriteHTKFeature(pOutFp, &pOut[j*nCoeffs], nCoeffs, swap, targetKind & PARAMKIND_C, pScale, pBias)) {
        return -1;
      }
    }
    return 0;
  }
  

  //***************************************************************************
  //***************************************************************************
  int 
  FeatureRepository::
  WriteHTKFeatures(
    FILE *  pOutFp,
    int     samplePeriod,
    int     targetKind,  
    bool    swap,
    Matrix<BaseFloat>&        rFeatureMatrix)
  {
    HtkHeader header;
    size_t i, j;
    FLOAT *p_scale = NULL;
    FLOAT *p_bias = NULL;
    size_t n_samples = rFeatureMatrix.Rows();
    size_t n_coeffs  = rFeatureMatrix.Cols();
    
    header.mNSamples = n_samples  + ((targetKind & PARAMKIND_C) ? 2 * sizeof(FLOAT_32) / sizeof(INT_16) : 0);
    header.mSamplePeriod = samplePeriod;
    header.mSampleSize = n_coeffs * ((targetKind & PARAMKIND_C) ?    sizeof(INT_16)   : sizeof(FLOAT_32));;
    header.mSampleKind = targetKind;
    
    WriteHTKHeader (pOutFp, header, swap);

    if(targetKind & PARAMKIND_C) {
      p_scale = (FLOAT*) malloc(n_coeffs * sizeof(FLOAT));
      p_bias = (FLOAT*)  malloc(n_coeffs * sizeof(FLOAT));
      if (p_scale == NULL || p_bias == NULL) Error("Insufficient memory");
      
      for(i = 0; i < n_coeffs; i++) {
        float xmin, xmax;
	xmin = xmax = rFeatureMatrix[0][i];

	for(j = 1; j < n_samples; j++) {
	  if(rFeatureMatrix[j][i] > xmax) xmax = rFeatureMatrix[j][i];
	  if(rFeatureMatrix[j][i] < xmin) xmin = rFeatureMatrix[j][i];
	}

	p_scale[i] = (2*32767) / (xmax - xmin);
        p_bias[i]  = p_scale[i] * (xmax + xmin) / 2;
      }

      if (WriteHTKFeature(pOutFp, p_scale, n_coeffs, swap, false, 0, 0)
      ||  WriteHTKFeature(pOutFp, p_bias,  n_coeffs, swap, false, 0, 0)) {
        return -1;
      }
    }

    for(j = 0; j < n_samples; j++) {
      if (WriteHTKFeature(pOutFp, rFeatureMatrix[j].pData(), n_coeffs, swap, targetKind & PARAMKIND_C, p_scale, p_bias)) {
        return -1;
      }
    }

    return 0;
  }
  
  //***************************************************************************
  //***************************************************************************


  bool 
  FeatureRepository::
  ReadGzipAsciiFeatures(const FileListElem& rFileNameRecord, Matrix<BaseFloat>& rFeatureMatrix)
  {
    //build the command
    std::string cmd("gunzip -c "); cmd += rFileNameRecord.Physical();

    //define buffer
    const int buf_size=262144;
    char buf[buf_size];
    char vbuf[2*buf_size];

   TIMER_START(mTim);      
    //open the pipe
    FILE* fp = popen(cmd.c_str(),"r");
    if(fp == NULL) {
      //2nd try...
      Warning(std::string("2nd try to open pipe: ")+cmd);
      sleep(5);
      fp = popen(cmd.c_str(),"r");
      if(fp == NULL) {
        KALDI_ERR << "Cannot open pipe: " << cmd;
      }
    }
    setvbuf(fp,vbuf,_IOFBF,2*buf_size);
   TIMER_END(mTim,mTimeOpen);

    //string will stay allocated across calls
    static std::string line; line.resize(0);

    //define matrix storage
    static int cols = 131072;
    std::list<std::vector<BaseFloat> > matrix(1);
    matrix.front().reserve(cols);

    //read all the lines to a vector
    int line_ctr=1;
    while(1) {
     TIMER_START(mTim);      
      if(NULL == fgets(buf,buf_size,fp)) break;
     TIMER_END(mTim,mTimeRead);
      
      line += buf;
      if(*(line.rbegin()) == '\n' || feof(fp)) {
        //parse the line of numbers
       TIMER_START(mTim);      
        const char* ptr = line.c_str();
        char* end;
        while(1) {
          //skip whitespace
          while(isspace(*ptr)) ptr++;
          if(*ptr == 0) break;
          //check that a number follows
          switch(*ptr) {
            case '0': case '1': case '2': case '3': case '4': 
            case '5': case '6': case '7': case '8': case '9':
            case '.': case '+': case '-': 
            break;
            default : KALDI_ERR << "A number was expected:" << ptr
                                << " reading from" << cmd; 
                      exit(1);
          }
          //read a number
          BaseFloat val = strtof(ptr,&end); ptr=end;
          matrix.back().push_back(val);
        }
       TIMER_END(mTim,mTimeNormalize);
        //we have the line of numbers, insert empty row to matrix
        if(matrix.back().size() > 0 && !feof(fp)) {
          matrix.push_back(std::vector<BaseFloat>());
          matrix.back().reserve(matrix.front().size());
        }
        //dispose the current line
        line.resize(0);//but stay allocated... 
        line_ctr++;
      }
    }
    if(matrix.back().size() == 0) matrix.pop_back();

    //get matrix dimensions
    int rows = matrix.size();
    /*int*/ cols = matrix.front().size();

    //define interators
    std::list<std::vector<BaseFloat> >::iterator it_r;
    std::vector<BaseFloat>::iterator it_c;

    //check that all lines have same size
    int i;
    for(i=0,it_r=matrix.begin(); it_r != matrix.end(); ++i,++it_r) {
      if(it_r->size() != cols) {
        KALDI_ERR << "All rows must have same dimension, 1st line cols: " << cols 
                  << ", " << i << "th line cols: " << it_r->size();
      }
    }

    //copy data to matrix
   TIMER_START(mTim);      
    rFeatureMatrix.Init(rows,cols);
    int r,c;
    for(r=0,it_r=matrix.begin(); it_r!=matrix.end(); ++r,++it_r) {
      for(c=0,it_c=it_r->begin(); it_c!=it_r->end(); ++c,++it_c) {
        rFeatureMatrix(r,c) = *it_c;
      }
    }
   TIMER_END(mTim,mTimeSeek);

    //close the pipe
    if(pclose(fp) == -1) {
      KALDI_ERR << "Cannot close pipe: " << cmd;
    }
    
    return true;
  }


  //***************************************************************************
  //***************************************************************************

} // namespace TNet
