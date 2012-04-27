#ifndef TNet_Common_h
#define TNet_Common_h

#include <cstdlib>
#include <string.h> // C string stuff like strcpy
#include <string>
#include <sstream>
#include <stdexcept>

/* Alignment of critical dynamic data structure
 *
 * Not all platforms support memalign so we provide a stk_memalign wrapper
 * void *stk_memalign( size_t align, size_t size, void **pp_orig )
 * *pp_orig is the pointer that has to be freed afterwards.
 */
#ifdef HAVE_POSIX_MEMALIGN
#  define stk_memalign(align,size,pp_orig) \
     ( !posix_memalign( pp_orig, align, size ) ? *(pp_orig) : NULL )
#  ifdef STK_MEMALIGN_MANUAL
#    undef STK_MEMALIGN_MANUAL
#  endif
#elif defined(HAVE_MEMALIGN)
   /* Some systems have memalign() but no declaration for it */
   //void * memalign( size_t align, size_t size );
#  define stk_memalign(align,size,pp_orig) \
     ( *(pp_orig) = memalign( align, size ) )
#  ifdef STK_MEMALIGN_MANUAL
#    undef STK_MEMALIGN_MANUAL
#  endif
#else /* We don't have any choice but to align manually */
#  define stk_memalign(align,size,pp_orig) \
     (( *(pp_orig) = malloc( size + align - 1 )) ? \
     (void *)( (((unsigned long)*(pp_orig)) + 15) & ~0xFUL ) : NULL )
#  define STK_MEMALIGN_MANUAL
#endif


#define swap8(a) { \
  char t=((char*)&a)[0]; ((char*)&a)[0]=((char*)&a)[7]; ((char*)&a)[7]=t;\
      t=((char*)&a)[1]; ((char*)&a)[1]=((char*)&a)[6]; ((char*)&a)[6]=t;\
      t=((char*)&a)[2]; ((char*)&a)[2]=((char*)&a)[5]; ((char*)&a)[5]=t;\
      t=((char*)&a)[3]; ((char*)&a)[3]=((char*)&a)[4]; ((char*)&a)[4]=t;}
#define swap4(a) { \
  char t=((char*)&a)[0]; ((char*)&a)[0]=((char*)&a)[3]; ((char*)&a)[3]=t;\
      t=((char*)&a)[1]; ((char*)&a)[1]=((char*)&a)[2]; ((char*)&a)[2]=t;}
#define swap2(a) { \
  char t=((char*)&a)[0]; ((char*)&a)[0]=((char*)&a)[1]; ((char*)&a)[1]=t;}


namespace TNet
{
  /** **************************************************************************
   ** **************************************************************************
   * @brief Aligns a number to a specified base
   * @param n Number of type @c _T to align
   * @return Aligned value of type @c _T
   */
  template<size_t _align, typename _T>
    inline _T 
    align(const _T n)
    {
      const _T x(_align - 1); 
      return (n + x) & ~(x);
    }


  /** 
   * @brief Returns true if architecture is big endian
   */
  bool 
  IsBigEndian();


  /** 
   * @brief Returns true if two numbers are close enough to each other
   * 
   * @param f1  First operand
   * @param f2  Second operand
   * @param nRounds Expected number of operations prior to this comparison
  */
  bool 
  CloseEnough(const float f1, const float f2, const float nRounds);
  

  /** 
   * @brief Returns true if two numbers are close enough to each other
   * 
   * @param f1  First operand
   * @param f2  Second operand
   * @param nRounds Expected number of operations prior to this comparison
  */
  bool 
  CloseEnough(const double f1, const double f2, const double nRounds);
  

  /** 
   * @brief Parses a HTK-style string into a C++ std::string readable
   * 
   * @param rIn  HTK input string
   * @param rOut output parsed string
   */
  void
  ParseHTKString(const std::string & rIn, std::string & rOut);

  
  /** 
   * @brief Synthesize new file name based on name, path, and extension
   * 
   * @param pOutFileName  full ouptut file name
   * @param pInFileName   file name
   * @param pOutDir       directory
   * @param pOutExt       extension
   */
  void    
  MakeHtkFileName(char *pOutFileName, const char* pInFileName, const char *pOutDir, 
      const char *pOutExt);
  

  /** 
   * @brief Removes the leading and trailing white chars
   *
   * @param rStr Refference to the string to be processed
   * @return Refference to the original string
   *
   * The white characters are determined by the @c WHITE_CHARS macro defined 
   * above.
   */
  std::string&
  Trim(std::string& rStr);


  char*
  StrToUpper(char* pStr);

  char* 
  ExpandHtkFilterCmd(const char *command, const char *filename, const char* pFilter);
  
  
  template <class T>
  std::string to_string(const T& val)
  {
    std::stringstream ss;
    ss << val;
    return ss.str();
  }
  
  inline void 
  ExpectKeyword(std::istream &i_stream, const char *kwd)
  {
     std::string token;
     i_stream >> token;
     if (token != kwd) {
       throw std::runtime_error(std::string(kwd) + " expected");
     }
  }
  
  extern const int MATRIX_IOS_FORMAT_IWORD;

  enum MatrixVectorIostreamControlBits {
    ACCUMULATE_INPUT = 1,
//  BINARY_OUTPUT    = 2
  };
  
  class MatrixVectorIostreamControl
  {
    public:
      MatrixVectorIostreamControl(enum MatrixVectorIostreamControlBits bitsToBeSet, bool valueToBeSet)
      : mBitsToBeSet(bitsToBeSet), mValueToBeSet(valueToBeSet) {}
      
      static long Flags(std::ios_base &rIos, enum MatrixVectorIostreamControlBits bits)
      { return rIos.iword(MATRIX_IOS_FORMAT_IWORD); }
      
      long mBitsToBeSet;
      bool mValueToBeSet;
            
      friend std::ostream & operator <<(std::ostream &rOs, const MatrixVectorIostreamControl modifier)
      {
        if(modifier.mValueToBeSet) {
          rOs.iword(MATRIX_IOS_FORMAT_IWORD) |= modifier.mBitsToBeSet;
        } else {
          rOs.iword(MATRIX_IOS_FORMAT_IWORD) &= ~modifier.mBitsToBeSet;
        }
        return rOs;
      }

      friend std::istream & operator >>(std::istream &rIs, const MatrixVectorIostreamControl modifier)
      {
        if(modifier.mValueToBeSet) {
          rIs.iword(MATRIX_IOS_FORMAT_IWORD) |= modifier.mBitsToBeSet;
        } else {
          rIs.iword(MATRIX_IOS_FORMAT_IWORD) &= ~modifier.mBitsToBeSet;
        }
        return rIs;
      }
  };
  
  
  

} // namespace TNet

#ifdef __ICC
#pragma warning (disable: 383) // ICPC remark we don't want.
#pragma warning (disable: 810) // ICPC remark we don't want.
#pragma warning (disable: 981) // ICPC remark we don't want.
#pragma warning (disable: 1418) // ICPC remark we don't want.
#pragma warning (disable: 444) // ICPC remark we don't want.
#pragma warning (disable: 869) // ICPC remark we don't want.
#pragma warning (disable: 1287) // ICPC remark we don't want.
#pragma warning (disable: 279) // ICPC remark we don't want.
#pragma warning (disable: 981) // ICPC remark we don't want.
#endif

//#ifdef CYGWIN
#if 1
#undef assert
#ifndef NDEBUG
#define assert(e)          ((e) ? (void)0 : assertf(__FILE__, __LINE__, #e))
#else
#define assert(e)         ((void)0)
#endif
void assertf(const char *c, int i, const char *msg); // Just make it possible to break into assert on gdb-- has some kind of bug on cygwin.
#else
#include <cassert>
#endif

#define assert_throw(e)          ((e) ? (void)0 : assertf_throw(__FILE__, __LINE__, #e))
void assertf_throw(const char *c, int i, const char *msg); 

#define DAN_STYLE_IO

#endif // ifndef TNet_Common_h

