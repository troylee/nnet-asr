#include <string>
#include <stdexcept>
#include <cmath>
#include <cfloat>
#include <cstdio>

#include "Common.h"
#include "MathAux.h"


/// Defines the white chars for string trimming
#if !defined(WHITE_CHARS)
#  define WHITE_CHARS " \t"
#endif

namespace TNet {

#include <ios>
  
  // Allocating stream variable used by stream modifier MatrixVectorIostreamControl
  const int MATRIX_IOS_FORMAT_IWORD = std::ios_base::xalloc();

  //***************************************************************************
  //***************************************************************************
  int getHTKstr(char *str)
  {
    char termChar = '\0';
    char *chrptr = str;
  
    while (std::isspace(*chrptr)) ++chrptr;
  
    if (*chrptr == '\'' || *chrptr == '"') {
      termChar = *chrptr;
      chrptr++;
    }
  
    for (; *chrptr; chrptr++) {
      if (*chrptr == '\'' || *chrptr == '"') {
        if (termChar == *chrptr) {
          termChar = '\0';
          chrptr++;
          break;
        }
      }
  
      if (std::isspace(*chrptr) && !termChar) {
        break;
      }
  
      if (*chrptr == '\\') {
        ++chrptr;
        if (*chrptr == '\0' || (*chrptr    >= '0' && *chrptr <= '7' &&
                              (*++chrptr  <  '0' || *chrptr >  '7' ||
                              *++chrptr  <  '0' || *chrptr >  '7'))) {
          return -1;
        }
  
        if (*chrptr  >= '0' && *chrptr  <= '7') {
          *chrptr = (char)((*chrptr - '0') + (chrptr[-1] - '0') * 8 + (chrptr[-2] - '0') * 64);
        }
      }
      *str++ = *chrptr;
    }
  
    if (termChar) {
      return -2;
    }
  
    *str = '\0';
  
    return 0;
  }
  

  //*****************************************************************************
  //*****************************************************************************
  void
  ParseHTKString(const std::string & rIn, std::string & rOut)
  {
    int ret_val;

    // the new string will be at most as long as the original, so we allocate
    // space
    char* new_str = new char[rIn.size() + 1];

    char* p_htk_str = new_str;

    strcpy(p_htk_str, rIn.c_str());
    ret_val = getHTKstr(p_htk_str);

    // call the function
    if (!ret_val) {
      rOut = p_htk_str;
    }

    delete [] new_str;

    if (ret_val) {
      throw std::runtime_error("Error parsing HTK string");
    }
  }

  

  //***************************************************************************
  //***************************************************************************
  bool 
  IsBigEndian()
  {
    int a = 1;
    return (bool) ((char *) &a)[0] != 1;
  }
  

  //***************************************************************************
  //***************************************************************************
  void 
  MakeHtkFileName(char* pOutFileName,  const char* inFileName,
               const char* out_dir, const char* out_ext)
  {
    const char* base_name;
    const char* bname_end = NULL;
    const char* chrptr;
  
    //  if (*inFileName == '*' && *++inFileName == '/') ++inFileName;
    
    // we don't do anything if file is stdin/out
    if (!strcmp(inFileName, "-"))
    {
      pOutFileName[0] = '-';
      pOutFileName[1] = '\0';
      return;
    }    
    
    base_name = strrchr(inFileName, '/');
    base_name = base_name != NULL ? base_name + 1 : inFileName;
    
    if (out_ext) bname_end = strrchr(base_name, '.');
    if (!bname_end) bname_end = base_name + strlen(base_name);
  
  
    if ((chrptr = strstr(inFileName, "/./")) != NULL) 
    {
      // what is in path after /./ serve as base name
      base_name = chrptr + 3;
    }
    /* else if (*inFileName != '/') 
    {
      // if inFileName isn't absolut path, don't forget directory structure
      base_name = inFileName;
    }*/
  
    *pOutFileName = '\0';
    if (out_dir) 
    {
      if (*out_dir) 
      {
        strcat(pOutFileName, out_dir);
        strcat(pOutFileName, "/");
      }
      strncat(pOutFileName, base_name, bname_end-base_name);
    } 
    else 
    {
      strncat(pOutFileName, inFileName, bname_end-inFileName);
    }
  
    if (out_ext && *out_ext) 
    {
      strcat(pOutFileName, ".");
      strcat(pOutFileName, out_ext);
    }
  }

  
  //****************************************************************************
  //****************************************************************************
  bool 
  CloseEnough(const float f1, const float f2, const float nRounds)
  {
    bool ret_val = (_ABS((f1 - f2) / (f2 == 0.0f ? 1.0f : f2))
        < (nRounds * FLT_EPSILON));

    return ret_val;
  } 

  
  //****************************************************************************
  //****************************************************************************
  bool 
  CloseEnough(const double f1, const double f2, const double nRounds)
  {
    bool ret_val = (_ABS((f1 - f2) / (f2 == 0.0 ? 1.0 : f2))
        < (nRounds * DBL_EPSILON));

    return ret_val;
  } 


  //****************************************************************************
  //****************************************************************************
  char* 
  ExpandHtkFilterCmd(const char *command, const char *filename, const char* pFilter)
  {

    char *out, *outend;
    const char *chrptr = command;
    int ndollars = 0;
    int fnlen = strlen(filename);

    while (*chrptr++) ndollars += (*chrptr ==  *pFilter);

    out = (char*) malloc(strlen(command) - ndollars + ndollars * fnlen + 1);

    outend = out;

    for (chrptr = command; *chrptr; chrptr++) {
      if (*chrptr ==  *pFilter) {
        strcpy(outend, filename);
        outend += fnlen;
      } else {
        *outend++ = *chrptr;
      }
    }
    *outend = '\0';
    return out;
  }

  //***************************************************************************
  //***************************************************************************
  char *
  StrToUpper(char *str)
  {
    char *chptr;
    for (chptr = str; *chptr; chptr++) {
      *chptr = (char)toupper(*chptr);
    }
    return str;
  }
  

  //**************************************************************************** 
  //**************************************************************************** 
  std::string&
  Trim(std::string& rStr)
  {
    // WHITE_CHARS is defined in common.h
    std::string::size_type pos = rStr.find_last_not_of(WHITE_CHARS);
    if(pos != std::string::npos) 
    {
      rStr.erase(pos + 1);
      pos = rStr.find_first_not_of(WHITE_CHARS);
      if(pos != std::string::npos) rStr.erase(0, pos);
    }
    else 
      rStr.erase(rStr.begin(), rStr.end());

    return rStr;
  }


} // namespace TNet

//#ifdef CYGWIN

void assertf(const char *c, int i, const char *msg){
  printf("Assertion \"%s\" failed: file \"%s\", line %d\n", msg?msg:"(null)", c?c:"(null)", i);
  abort();
}


void assertf_throw(const char *c, int i, const char *msg){
  char buf[2000];
  snprintf(buf, 1999, "Assertion \"%s\" failed, throwing exception: file \"%s\", line %d\n", msg?msg:"(null)", c?c:"(null)", i);
  throw std::runtime_error((std::string)buf);
}
//#endif
