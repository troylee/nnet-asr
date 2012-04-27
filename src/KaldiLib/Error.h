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

/** @file Error.h
 *  This header defines several types and functions relating to the
 *  handling of exceptions in STK.
 */
 
#ifndef TNET_Error_h
#define TNET_Error_h

#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>

#include <cstdlib>
#include <execinfo.h>

// THESE MACROS TERRIBLY CLASH WITH STK!!!!
// WE MUST USE SAME MACROS!
//
//#define Error(msg) _Error_(__func__, __FILE__, __LINE__, msg)
//#define Warning(msg) _Warning_(__func__, __FILE__, __LINE__, msg)
//#define TraceLog(msg) _TraceLog_(__func__, __FILE__, __LINE__, msg)
//

#ifndef Error
  #define Error(...) _Error_(__func__, __FILE__, __LINE__, __VA_ARGS__)
#endif
#ifndef Warning
  #define Warning(...) _Warning_(__func__, __FILE__, __LINE__, __VA_ARGS__)
#endif
#ifndef TraceLog
  #define TraceLog(...) _TraceLog_(__func__, __FILE__, __LINE__, __VA_ARGS__)
#endif

namespace TNet {
  


  /** MyException
   * Custom exception class, gets the stacktrace
   */
  class MyException 
    : public std::runtime_error
  {
    public:
      explicit MyException(const std::string& what_arg) throw();
      virtual ~MyException() throw();

      const char* what() const throw() 
      { return mWhat.c_str(); }

    private:
      std::string mWhat;
  };

  /** 
   * MyException:: implemenatation
   */
  inline
  MyException::
  MyException(const std::string& what_arg) throw()
    : std::runtime_error(what_arg)
  {
    mWhat = what_arg;
    mWhat += "\nTHE STACKTRACE INSIDE MyException OBJECT IS:\n";
    
    void *array[10];
    size_t size;
    char **strings;
    size_t i;

    size = backtrace (array, 10);
    strings = backtrace_symbols (array, size);
    
    //<< 0th string is the MyException ctor, so ignore and start by 1
    for (i = 1; i < size; i++) { 
      mWhat += strings[i];
      mWhat += "\n";
    }

    free (strings);
  }


  inline
  MyException::
  ~MyException() throw()
  { } 



  /**
   *  @brief Error throwing function (with backtrace)
   */
  inline void 
  _Error_(const char *func, const char *file, int line, const std::string &msg)
  {
     std::stringstream ss;
     ss << "ERROR (" << func << ':' << file  << ':' << line << ") " << msg;
     throw MyException(ss.str());
  }
  
  /**
   *  @brief Warning handling function
   */
  inline void 
  _Warning_(const char *func, const char *file, int line, const std::string &msg)
  {
	std::cout << "WARNING (" << func << ':' << file  << ':' << line << ") " << msg << std::endl;
  }

  inline void 
  _TraceLog_(const char *func, const char *file, int line, const std::string &msg)
  {
	std::cout << "INFO (" << func << ':' << file  << ':' << line << ") " << msg << std::endl;
  }

  /**
   * New kaldi error handling:
   *
   * class KaldiErrorMessage is invoked from the KALDI_ERROR macro.
   * The destructor throws an exception.
   */
  class KaldiErrorMessage {
   public:
    KaldiErrorMessage(const char *func, const char *file, int line) {
      this->stream() << "ERROR (" 
                     << func << "():"
                     << file << ':' << line << ") ";
    }
    inline std::ostream &stream() { return ss; }
    ~KaldiErrorMessage() { throw MyException(ss.str()); }
   private:
    std::ostringstream ss;
  };
  #define KALDI_ERR TNet::KaldiErrorMessage(__func__, __FILE__, __LINE__).stream() 



} // namespace TNet

//#define TNET_Error_h
#endif
