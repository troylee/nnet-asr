#ifndef TNet_MathAux_h
#define TNet_MathAux_h

#include <cmath>


#if !defined(SQR)
# define SQR(x) ((x) * (x))
#endif


#if !defined(LOG_0)
# define LOG_0 (-1.0e10)
#endif

#if !defined(LOG_MIN)
# define LOG_MIN   (0.5 * LOG_0)
#endif


#ifndef DBL_EPSILON
#define DBL_EPSILON 2.2204460492503131e-16
#endif


#ifndef M_PI
#  define M_PI 3.1415926535897932384626433832795
#endif

#define M_LOG_2PI 1.8378770664093454835606594728112


#if DOUBLEPRECISION
#  define FLOAT double
#  define EPSILON DBL_EPSILON
#  define FLOAT_FMT "%lg"
#  define swapFLOAT swap8
#  define _ABS  fabs
#  define _COS  cos
#  define _EXP  exp
#  define _LOG  log
#  define _SQRT sqrt
#else
#  define FLOAT float
#  define EPSILON FLT_EPSILON
#  define FLOAT_FMT "%g"
#  define swapFLOAT swap4
#  define _ABS  fabsf
#  define _COS  cosf
#  define _EXP  expf
#  define _LOG  logf
#  define _SQRT sqrtf
#endif

namespace TNet
{
  inline float frand(){ // random between 0 and 1.
	return (float(rand()) + 1.0f) / (float(RAND_MAX)+2.0f);
  }
  inline float gauss_rand(){
	return _SQRT( -2.0f * _LOG(frand()) ) * _COS(2.0f*float(M_PI)*frand());
  }
  
  static const double gMinLogDiff = log(DBL_EPSILON);
  
  //***************************************************************************
  //***************************************************************************
  inline double
  LogAdd(double x, double y)
  {
    double diff;
  
    if (x < y) {
      diff = x - y;
      x = y;
    } else {
      diff = y - x;
    }
  
    double res;
    if (x >= LOG_MIN) {
      if (diff >= gMinLogDiff) {
        res = x + log(1.0 + exp(diff));
      } else {
        res = x;
      }
    } else {
      res = LOG_0;
    }
    return res;
  } 


  //***************************************************************************
  //***************************************************************************
  inline double
  LogSub(double x, double y) // returns exp(x) - exp(y).  Throws exception if y>=x.
  {

    if(y >= x){
      if(y==x)  return LOG_0;
      else throw std::runtime_error("LogSub: cannot subtract a larger from a smaller number.");
    }

    double diff = y - x;  // Will be negative.
    
    double res = x + log(1.0 - exp(diff));

    if(res != res) // test for res==NaN.. could happen if diff ~0.0, so 1.0-exp(diff) == 0.0 to machine precision.
      res = LOG_0;
    return res;
  } 

} // namespace TNet


#endif
