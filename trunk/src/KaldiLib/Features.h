//
// C++ Interface: %{MODULE}
//
// Description: 
//
//
// Author: %{AUTHOR} <%{EMAIL}>, (C) %{YEAR}
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef TNet_Features_h
#define TNet_Features_h

//*****************************************************************************
//*****************************************************************************
// Standard includes
//
#include <list>
#include <queue>
#include <string>


//*****************************************************************************
//*****************************************************************************
// Specific includes
//
#include "Common.h"
#include "Matrix.h"
#include "StkStream.h"
#include "Types.h"
#include "Timer.h"



// we need these for reading and writing
#define UINT_16  unsigned short
#define UINT_32  unsigned
#define INT_16   short
#define INT_32   int
#define FLOAT_32 float
#define DOUBLE_64 double


#define PARAMKIND_WAVEFORM  0
#define PARAMKIND_LPC       1
#define PARAMKIND_LPREFC    2
#define PARAMKIND_LPCEPSTRA 3
#define PARAMKIND_LPDELCEP  4
#define PARAMKIND_IREFC     5
#define PARAMKIND_MFCC      6
#define PARAMKIND_FBANK     7
#define PARAMKIND_MELSPEC   8
#define PARAMKIND_USER      9
#define PARAMKIND_DISCRETE 10
#define PARAMKIND_PLP      11
#define PARAMKIND_ANON     12

#define PARAMKIND_E   0000100 /// has energy
#define PARAMKIND_N   0000200 /// absolute energy suppressed
#define PARAMKIND_D   0000400 /// has delta coefficients
#define PARAMKIND_A   0001000 /// has acceleration coefficients
#define PARAMKIND_C   0002000 /// is compressed
#define PARAMKIND_Z   0004000 /// has zero mean static coef.
#define PARAMKIND_K   0010000 /// has CRC checksum
#define PARAMKIND_0   0020000 /// has 0'th cepstral coef.
#define PARAMKIND_V   0040000 /// has VQ codebook index
#define PARAMKIND_T   0100000 /// has triple delta coefficients


//*****************************************************************************
//*****************************************************************************
// Code ...
//

namespace TNet
{

  /** **************************************************************************
   ** **************************************************************************
   */
  class FileListElem
  {
  private:
    std::string         mLogical;     ///< Logical file name representation
    std::string         mPhysical;    ///< Pysical file name representation
    float               mWeight;
    
  public:
    FileListElem(const std::string & rFileName);
    ~FileListElem() {}
    
    const std::string &
    Logical() const { return mLogical; }

    const std::string &
    Physical() const { return mPhysical; }

    const float&
    Weight() const { return mWeight; }
  };

  /** *************************************************************************
   * @brief 
   */
  class FeatureRepository
  {
  public:
    /** 
     * @brief HTK parameter file header (see HTK manual)
     */
    struct HtkHeader
    {
      int   mNSamples;
      int   mSamplePeriod;
      short mSampleSize;
      short mSampleKind;

      HtkHeader() 
       : mNSamples(0),mSamplePeriod(100000),mSampleSize(0),mSampleKind(12)
      { }
    };


    /**
     *     @brief Extension of the HTK header
     */
    struct HtkHeaderExt
    {
      int mHeaderSize;
      int mVersion;
      int mSampSize;
    };


    /** 
     * @brief Normalization file type
     */
    enum CNFileType
    {
      CNF_Mean,
      CNF_Variance,
      CNF_VarScale
    };


    static int     
    ReadParmKind(const char *pStr, bool checkBrackets);

    static int     
    ParmKind2Str(unsigned parmKind, char *pOutstr);

    static void 
    ReadCepsNormFile(
        const char*   pFileName,
        char**        lastFile,
        BaseFloat**   vecBuff,
        int           sampleKind,
        CNFileType    type,
        int           coefs);

    static const char mpParmKindNames[13][16];

  
  
  //////////////////////////////////////////////////////////////////////////////
  //  PUBLIC SECTION
  //////////////////////////////////////////////////////////////////////////////
  public:
    /// Iterates through the list of feature file records
    typedef   std::list<FileListElem>::iterator  ListIterator;

    // some params for loading features
    bool                        mSwapFeatures;
    int                         mStartFrameExt;
    int                         mEndFrameExt;
    int                         mTargetKind;
    int                         mDerivOrder;
    int*                        mDerivWinLengths;
    const char*                 mpCvgFile;
    //:TODO: get rid of these
    const char*                 mpCmnPath;
    const char*                 mpCmnMask;
    const char*                 mpCvnPath;
    const char*                 mpCvnMask;

    int                         mTrace;
    
    
    // Constructors and destructors
    /**
     * @brief Default constructor that creates an empty repository
     */
    FeatureRepository() : mDerivWinLengths(NULL), mpCvgFile(NULL), 
       mpCmnPath(NULL), mpCmnMask(NULL), mpCvnPath(NULL), mpCvnMask(NULL),
       mTrace(0),
       mpLastFileName(NULL), mLastFileName(""), mpLastCmnFile (NULL), 
       mpLastCvnFile (NULL), mpLastCvgFile (NULL), mpCmn(NULL), 
       mpCvn(NULL), mpCvg(NULL), mpA(NULL), mpB(NULL),
       mTimeOpen(0), mTimeSeek(0), mTimeRead(0), mTimeNormalize(0) 
    { 
      mInputQueueIterator        = mInputQueue.end();
    }

    /**
     * @brief Copy constructor which copies filled repository
     */
    FeatureRepository(const FeatureRepository& ori)
     : mDerivWinLengths(NULL), mpCvgFile(NULL), 
       mpCmnPath(NULL), mpCmnMask(NULL), mpCvnPath(NULL), mpCvnMask(NULL),
       mTrace(0),
       mpLastFileName(NULL), mLastFileName(""), mpLastCmnFile (NULL), 
       mpLastCvnFile (NULL), mpLastCvgFile (NULL), mpCmn(NULL), 
       mpCvn(NULL), mpCvg(NULL), mpA(NULL), mpB(NULL),
       mTimeOpen(0), mTimeSeek(0), mTimeRead(0), mTimeNormalize(0) 
    {
      //copy all the data from the input queue
      mInputQueue = ori.mInputQueue;

      //initialize like the original
      Init(
        ori.mSwapFeatures,
        ori.mStartFrameExt,
        ori.mEndFrameExt,
        ori.mTargetKind,
        ori.mDerivOrder,
        ori.mDerivWinLengths,
        ori.mpCmnPath,
        ori.mpCmnMask,
        ori.mpCvnPath,
        ori.mpCvnMask,
        ori.mpCvgFile);
     
      //set on the end 
      mInputQueueIterator        = mInputQueue.end(); 
      //copy default header values
      mHeader = ori.mHeader;
    }


    /**
     * @brief Destroys the repository
     */
    ~FeatureRepository()
    {
      if (NULL != mpA) {
        free(mpA);
      }

      if (NULL != mpB) {
        free(mpB);
      }
      //remove all entries
      mInputQueue.clear();

      if(mTrace&4) {
        std::cout << "[FeatureRepository -- open:" << mTimeOpen << "s seek:" << mTimeSeek << "s read:" << mTimeRead << "s normalize:" << mTimeNormalize << "s]\n";
      }

    }


    /**
     * @brief Initializes the object using the given parameters
     *
     * @param swap          Boolean value specifies whether to swap bytes 
     *                      when reading file or not. 
     * @param extLeft       Features read from file are extended with extLeft 
     *                      initial frames. Normally, these frames are 
     *                      repetitions of the first feature frame in file 
     *                      (with its derivative, if derivatives are preset in
     *                      the file). However, if segment of feature frames 
     *                      is extracted according to range specification, the 
     *                      true feature frames from beyond the segment boundary
     *                      are used, wherever it is possible. Note that value 
     *                      of extLeft can be also negative. In such case
     *                      corresponding number of initial frames is discarded. 
     * @param extRight      The paramerer is complementary to parameter extLeft 
     *                      and has obvious meaning. (Controls extensions over
     *                      the last frame, last frame from file is repeated 
     *                      only if necessary).
     * @param targetKind    The parameters is used to check whether 
     *                      pHeader->mSampleKind match to requited targetKind 
     *                      and to control suppression of 0'th cepstral or 
     *                      energy coefficients accorging to modifiers _E, _0, 
     *                      and _N. Modifiers _D, _A and _T are ignored; 
     *                      Computation of derivatives is controled by parameters
     *                      derivOrder and derivWinLen. Value PARAMKIND_ANON 
     *                      ensures that function do not result in targetKind 
     *                      mismatch error and cause no _E or _0 suppression.
     * @param derivOrder    Final features will be augmented with their 
     *                      derivatives up to 'derivOrder' order. If 'derivOrder'
     *                      is negative value, no new derivatives are appended 
     *                      and derivatives that already present in feature file
     *                      are preserved.  Straight features are considered 
     *                      to be of zero order. If some derivatives are already 
     *                      present in feature file, these are not computed 
     *                      again, only higher order derivatives are appended 
     *                      if required. Note, that HTK feature file cannot 
     *                      contain higher order derivatives (e.g. double delta)
     *                      without containing lower ones (e.g. delta). 
     *                      Derivative present in feature file that are of 
     *                      higher order than is required are discarded.  
     *                      Derivatives are computed in the final stage from 
     *                      (extracted segment of) feature frames possibly 
     *                      extended by repeated frames. Derivatives are 
     *                      computed using the same formula that is employed 
     *                      also by HTK tools. Lengths of windows used for 
     *                      computation of derivatives are passed in parameter 
     *                      derivWinLen. To compute derivatives for frames close 
     *                      to boundaries, frames before the first and after the 
     *                      last frame (of the extracted segment) are considered 
     *                      to be (yet another) repetitions of the first and the 
     *                      last frame, respectively. If the segment of frames 
     *                      is extracted according to range specification and 
     *                      parameters extLeft and extLeft are set to zero, the 
     *                      first and the last frames of the segment are 
     *                      considered to be repeated, eventough the true feature
     *                      frames from beyond the segment boundary can be
     *                      available in the file. Therefore, segment extracted 
     *                      from features that were before augmented with 
     *                      derivatives will differ 
     *                      from the same segment augmented with derivatives by 
     *                      this function. Difference will be of course only on 
     *                      boundaries and only in derivatives. This "incorrect" 
     *                      behavior was chosen to fully simulate behavior of 
     *                      HTK tools. To obtain more correct computation of 
     *                      derivatives, use parameters extLeft and extRight, 
     *                      which correctly extend segment with the true frames 
     *                      (if possible) and in resulting feature matrix ignore 
     *                      first extLeft and last extRight frames. For this 
     *                      purpose, both extLeft and extRight should be set to 
     *                      sum of all values in the array derivWinLen.
     * @param pDerivWinLen  Array of size derivOrder specifying lengths of 
     *                      windows used for computation of derivatives. 
     *                      Individual values represents one side context 
     *                      used in the computation. The each window length is 
     *                      therefore twice the value from array plus one. 
     *                      Value at index zero specify window length for first 
     *                      order derivatives (delta), higher indices 
     *                      corresponds to higher order derivatives.
     * @param pCmnPath      Cepstral mean normalization path
     * @param pCmnMask      Cepstral mean normalization mask
     * @param pCvnPath      Cepstral variance normalization path
     * @param pCvnMask      Cepstral variance normalization mask
     * @param pCvgFile      Global variance file to be parsed
     *
     * The given parameters are necessary for propper feature extraction 
     */
    void
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
        const char*           pCvgFile);
   
    
    void Trace(int trace)
    { mTrace = trace; } 
        
    /** 
     * @brief Returns a refference to the current file header
     */
    const HtkHeader&
    CurrentHeader() const 
    { return mHeader; }

    /** 
     * @brief Returns a refference to the current file header
     */
    const HtkHeaderExt&
    CurrentHeaderExt() const 
    { return mHeaderExt; }

    /**
     * @brief Returns the current file details
     *
     * @return Refference to a class @c FileListElem
     *
     * Logical and physical file names are stored in @c FileListElem class
     */
    const std::list<FileListElem>::iterator&
    pCurrentRecord() const
    { return mInputQueueIterator; }


    /**
     * @brief Returns the following file details
     *
     * @return Refference to a class @c FileListElem
     *
     * Logical and physical file names are stored in @c FileListElem class
     */
    const std::list<FileListElem>::iterator&
    pFollowingRecord() const
    { return mInputQueueIterator; }


    void
    Rewind()
    { mInputQueueIterator = mInputQueue.begin(); }
    
    
    /**
     * @brief Adds a single feature file to the repository
     * @param rFileName file to read features from
     */
    void
    AddFile(const std::string & rFileName);
    

    /**
     * @brief Adds a list of feature files to the repository
     * @param rFileName feature list file to read from
     */
    void
    AddFileList(const char* pFileName, const char* pFilter = "");
  
    
    const FileListElem&
    Current() const
    { return *mInputQueueIterator; }

    
    /** 
     * @brief Moves to the next record
     */
    void
    MoveNext();
    
    /**
     * @brief Reads full feature matrix from a feature file
     * @param rMatrix matrix to be created and filled with read data
     * @return number of successfully read feature vectors
     */
    bool
    ReadFullMatrix(Matrix<BaseFloat>& rMatrix); 
    
    bool
    WriteFeatureMatrix(const Matrix<BaseFloat>& rMatrix, const std::string& filename, int targetKind, int samplePeriod);
    
    size_t
    QueueSize() const {return mInputQueue.size(); }

    /**
     * @brief Reads feature vectors from a feature file
     * @param rMatrix matrix to be (only!) filled with read data. 
     * @return number of successfully read feature vectors
     * 
     * The function tries to fill @c pMatrix with feature vectors comming from
     * the current stream. If there are less vectors left in the stream, 
     * they are used and true number of successfuly read vectors is returned.
     */
    int
    ReadPartialMatrix(Matrix<BaseFloat>& rMatrix);    
    
    /** 
     * @brief Filters the records of this repository based on HTK logical name
     * masking. If pFilter equals to NULL, all source repository entries are
     * coppied to rOut repository.
     * 
     * @param pFilter HTK mask that defines the filter
     * @param pValue Filter value
     * @param rOut Reference to the new FeatureRepository which will be filled
     * with the matching records
     */
    void
    HtkFilter(const char* pFilter, const char* pValue, FeatureRepository& rOut);


    /** 
     * @brief Filters the records of this repository based on HTK logical name
     * masking and returns list of unique names. If pFilter equals to NULL, 
     * single name "default" is returned.
     * 
     * @param pFilter HTK mask that defines the filter
     * @param rOut Reference to the list of results (std::list< std::string >)
     */
    void
    HtkSelection(const char* pFilter, std::list< std::string >& rOut);


    /**
     * @brief Returns true if there are no feature files left on input
     */
    bool
    EndOfList() const 
    { return mInputQueueIterator == mInputQueue.end(); }

    const std::string&
    CurrentIndexFileName() const
    { return mCurrentIndexFileName; }
    
    friend
    void
    AddFileListToFeatureRepositories(
      const char* pFileName,
      const char* pFilter,
      std::queue<FeatureRepository *> &featureRepositoryList);


////////////////////////////////////////////////////////////////////////////////
//  PRIVATE SECTION
////////////////////////////////////////////////////////////////////////////////
  private:
    /// List (queue) of input feature files
    std::list<FileListElem>             mInputQueue;
    std::list<FileListElem>::iterator   mInputQueueIterator;
    
    std::string                         mCurrentIndexFileName;
    std::string                         mCurrentIndexFileDir;
    std::string                         mCurrentIndexFileExt;

    /// current stream
    IStkStream                  mStream;
      
    // stores feature file's HTK header
    HtkHeader                   mHeader;
    HtkHeaderExt                mHeaderExt;


    // this group of variables serve for working withthe same physical
    // file name more than once
    char*                       mpLastFileName;
    std::string                 mLastFileName;
    char*                       mpLastCmnFile;
    char*                       mpLastCvnFile;
    char*                       mpLastCvgFile;
    BaseFloat*                      mpCmn;
    BaseFloat*                      mpCvn;
    BaseFloat*                      mpCvg;
    HtkHeader                   mLastHeader;
    BaseFloat*                      mpA;
    BaseFloat*                      mpB;



    Timer mTim;
    double mTimeOpen;
    double mTimeSeek;
    double mTimeRead;
    double mTimeNormalize;


    // Reads HTK feature file header
    int 
    ReadHTKHeader();

    int 
    ReadHTKFeature(BaseFloat*    pIn, 
      size_t    feaLen, 
      bool      decompress, 
      BaseFloat*    pScale, 
      BaseFloat*    pBias);

    
    bool 
    ReadHTKFeatures(const std::string& rFileName, Matrix<BaseFloat>& rFeatureMatrix);
    
    bool 
    ReadHTKFeatures(const FileListElem& rFileNameRecord, Matrix<BaseFloat>& rFeatureMatrix);


    int 
    WriteHTKHeader  (FILE* fp_out, HtkHeader header, bool swap);

    int 
    WriteHTKFeature (FILE* fp_out, FLOAT *out, size_t fea_len, bool swap, bool compress, FLOAT* pScale, FLOAT* pBias);

    int 
    WriteHTKFeatures(FILE* pOutFp, FLOAT * pOut, int nCoeffs, int nSamples, int samplePeriod, int targetKind, bool swap);

    int 
    WriteHTKFeatures(
      FILE *  pOutFp,
      int     samplePeriod,
      int     targetKind,  
      bool    swap,
      Matrix<BaseFloat>& rFeatureMatrix
    );

    bool 
    ReadGzipAsciiFeatures(const FileListElem& rFileNameRecord, Matrix<BaseFloat>& rFeatureMatrix);

  }; // class FeatureStream

} //namespace TNet

#endif // TNet_Features_h
