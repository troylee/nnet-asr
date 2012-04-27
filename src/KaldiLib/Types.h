#ifndef TNet_Types_h
#define TNet_Types_h

#ifdef HAVE_ATLAS
extern "C"{
  #include <cblas.h>
  #include <clapack.h>
}
#endif


namespace TNet 
{
  // TYPEDEFS ..................................................................
#if DOUBLEPRECISION
  typedef double  BaseFloat;
#else
  typedef float   BaseFloat;
#endif

#ifndef UINT_16
  typedef unsigned short  UINT_16   ;
  typedef unsigned        UINT_32   ;
  typedef short           INT_16    ;
  typedef int             INT_32    ;
  typedef float           FLOAT_32  ;
  typedef double          DOUBLE_64 ;
#endif



  // ...........................................................................
  // The following declaration assumes that SSE instructions are enabled
  // and that we are using GNU C/C++ compiler, which defines the __attribute__ 
  // notation.
  //
  // ENABLE_SSE is defined in <config.h>. Its value depends on options given
  // in the configure phase of builidng the library
#if defined(__GNUC__ )
  // vector of four single floats
  typedef float  v4sf __attribute__((vector_size(16))); 
  // vector of two single doubles
  typedef double v2sd __attribute__((vector_size(16))); 

  typedef BaseFloat BaseFloat16Aligned __attribute__((aligned(16))) ;

  typedef union 
  {
    v4sf    v;
    float   f[4];
  } f4vector; 

  typedef union 
  {
    v2sd    v;
    double  f[2];
  } d2vector; 
#endif // ENABLE_SSE && defined(__GNUC__ )



  typedef enum
  {
#ifdef HAVE_ATLAS
    TRANS    = CblasTrans,
    NO_TRANS = CblasNoTrans
#else
    TRANS    = 'T',
    NO_TRANS = 'N'
#endif
  } MatrixTrasposeType;



} // namespace TNet

#endif // #ifndef TNet_Types_h

