#ifndef Timer_h
#define Timer_h

#include "Error.h"
#include <sstream>



#if defined(_WIN32) || defined(MINGW)

# include <windows.h>

namespace TNet
{
  class Timer {
  public:
    void 
    Start(void)
    {
      static int first = 1;

      if(first) {
              QueryPerformanceFrequency(&mFreq);
              first = 0;
      }
      QueryPerformanceCounter(&mTStart);
    }

    void 
    End(void)
    { QueryPerformanceCounter(&mTEnd); }

    double 
    Val()
    {
      return ((double)mTEnd.QuadPart - (double)mTStart.QuadPart) / 
        ((double)mFreq.QuadPart);
    }

  private:
    LARGE_INTEGER mTStart;
    LARGE_INTEGER mTEnd;
    LARGE_INTEGER mFreq;
  };
}

#else

# include <sys/time.h>
# include <unistd.h>

namespace TNet
{
  class Timer 
  {
  public:
    void 
    Start()
    { gettimeofday(&this->mTStart, &mTz); }

    void 
    End()
    { gettimeofday(&mTEnd,&mTz); }

    double 
    Val()
    {
      double t1, t2;

      t1 =  (double)mTStart.tv_sec + (double)mTStart.tv_usec/(1000*1000);
      t2 =  (double)mTEnd.tv_sec + (double)mTEnd.tv_usec/(1000*1000);
      return t2-t1;
    }

  private:
    struct timeval mTStart;
    struct timeval mTEnd;
    struct timezone mTz;
  };
}

#endif







///////////////////////////////////////////////////////////////
// Macros for adding the time intervals to time accumulator
#if PROFILING==1
#  define TIMER_START(timer) timer.Start()
#  define TIMER_END(timer,sum) timer.End(); sum += timer.Val()
#else
#  define TIMER_START(timer) 
#  define TIMER_END(timer,sum) 
#endif

#endif



