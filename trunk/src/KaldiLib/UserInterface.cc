#include <stdexcept>
#include <sstream>
#include <stdarg.h>

#include "UserInterface.h"
#include "StkStream.h"
#include "Features.h"

namespace TNet
{
  //***************************************************************************
  //***************************************************************************
  int 
  npercents(const char *str)
  {
    int ret = 0;
    while (*str) if (*str++ == '%') ret++;
    return ret;
  }
  

  //***************************************************************************
  //***************************************************************************
  void 
  UserInterface::
  ReadConfig(const char *file_name)
  {
    std::string   line_buf;
    std::string::iterator chptr;
    std::string   key;
    std::string   value;
    std::ostringstream ss;
    int           line_no = 0;
    IStkStream    i_stream;


    i_stream.open(file_name, std::ios::binary);
    if (!i_stream.good()) {
      throw std::runtime_error(std::string("Cannot open input config file ")
          + file_name);
    }
    i_stream >> std::ws;

    while (!i_stream.eof()) {
      size_t i_pos;

      // read line
      std::getline(i_stream, line_buf);
      i_stream >> std::ws;

      if (i_stream.fail()) {
        throw std::runtime_error(std::string("Error reading (") 
            + file_name + ":" + (ss << line_no,ss).str() + ")");
      }

      // increase line counter
      line_no++;

      // cut comments
      if (std::string::npos != (i_pos = line_buf.find('#'))) {
        line_buf.erase(i_pos);
      }

      // cut leading and trailing spaces
      Trim(line_buf);

      // if empty line, then skip it
      if (0 == line_buf.length()) {
        continue;
      }
  
      // line = line_buf.c_str();
      // chptr = parptr;

      chptr = line_buf.begin();

      for (;;) {
        // Replace speces by '_', which is removed in InsertConfigParam
        while (isalnum(*chptr) || *chptr == '_' || *chptr == '-') {
          chptr++;
        }

        while (std::isspace(*chptr)) {
          *chptr = '_';
          chptr++;
        }

        if (*chptr != ':') {
          break;
        }

        chptr++;

        while (std::isspace(*chptr)) {
          *chptr = '_';
          chptr++;
        }
      }
      
      if (*chptr != '=') {
        throw std::runtime_error(std::string("Character '=' expected (") 
            + file_name + ":" + (ss.str(""),ss<<line_no,ss).str() + ")");
      }

      key.assign(line_buf.begin(), chptr);

      chptr++;

      value.assign(chptr, line_buf.end());

      ParseHTKString(value, value);
      InsertConfigParam(key.c_str(), value.c_str(), 'C');
    }
  
    i_stream.close();
  }
  

  //***************************************************************************
  //***************************************************************************
  void 
  UserInterface::
  InsertConfigParam(const char *pParamName, const char *value, int optionChar) 
  {
    std::string key(pParamName);
    std::string::iterator i_key = key.begin();

    while (i_key != key.end()) {
      if (*i_key == '-' || *i_key == '_') {
        i_key = key.erase(i_key);
      }
      else {
        *i_key = toupper(*i_key);
        i_key ++;
      }
    }
  
    mMap[key].mValue  = value;
    mMap[key].mRead   = false;
    mMap[key].mOption = optionChar;
  }
  
  //***************************************************************************
  //***************************************************************************
  int 
  UserInterface::
  ParseOptions(
    int             argc,
    char*           argv[],
    const char*     pOptionMapping,
    const char*     pToolName)
  {
    int          i;
    int          opt = '?';
    int          optind;
    bool         option_must_follow = false;
    char         param[1024];
    char*        value;
    const char*  optfmt;
    const char*  optarg;
    char*        chptr;
    char*        bptr;
    char         tstr[4] = " -?";
    unsigned long long option_mask = 0;
    std::ostringstream ss;
  
    #define MARK_OPTION(ch) {if (isalpha(ch)) option_mask |= 1ULL << ((ch) - 'A');}
    #define OPTION_MARK(ch) (isalpha(ch) && ((1ULL << ((ch) - 'A')) & option_mask))
    #define IS_OPTION(str) ((str)[0] == '-' && (isalpha((str)[1]) || (str)[1] == '-'))
 
    //search for the -A param
    for (optind = 1; optind < argc; optind++) {
      // we found "--", no -A
      if (!strcmp(argv[optind], "--")) {
        break;
      }

      //repeat till we find -A
      if (argv[optind][0] != '-' || argv[optind][1] != 'A') {
        continue;
      }

      // just "-A" form
      if (argv[optind][2] != '\0') {
        throw std::runtime_error(std::string("Unexpected argument '") 
            + (argv[optind] + 2) + "' after option '-A'");
      }

      for (i=0; i < argc; i++) {
        // display all params
        if(strchr(argv[i], ' ') || strchr(argv[i], '*')) 
          std::cout << '\'' << argv[i] << '\'' << " ";
        else std::cout << argv[i] << " ";
      }

      std::cout << std::endl;

      break;
    }

    for (optind = 1; optind < argc; optind++) {
      // find the '-C?' parameter (possible two configs)
      if (!strcmp(argv[optind], "--")) break;
      if (argv[optind][0] != '-' || argv[optind][1] != 'C') continue;
      if (argv[optind][2] != '\0') {
        ReadConfig(argv[optind] + 2);
      } else if (optind+1 < argc && !IS_OPTION(argv[optind+1])) {
        ReadConfig(argv[++optind]);
      } else {
        throw std::runtime_error("Config file name expected after option '-C'");
      }
    }

    for (optind = 1; optind < argc; optind++) {
      if (!strcmp(argv[optind], "--")) break;
      if (argv[optind][0] != '-' || argv[optind][1] != '-') continue;

      bptr = new char[strlen(pToolName) + strlen(argv[optind]+2) + 2];
      strcat(strcat(strcpy(bptr, pToolName), ":"), argv[optind]+2);
      value = strchr(bptr, '=');
      if (!value) {
        throw std::runtime_error(std::string("Character '=' expected after option '")
            + argv[optind] + "'");
      }

      *value++ = '\0';
      
      InsertConfigParam(bptr, value /*? value : "TRUE"*/, '-');
      delete [] bptr;
    }

    for (optind = 1; optind < argc && IS_OPTION(argv[optind]); optind++) {
      option_must_follow = false;
      tstr[2] = opt = argv[optind][1];
      optarg = argv[optind][2] != '\0' ? argv[optind] + 2 : NULL;
  
      if (opt == '-' && !optarg) {    // '--' terminates the option list
        return optind+1;
      }
      if (opt == 'C' || opt == '-') { // C, A and long options have been already processed
        if (!optarg) optind++;
        continue;
      }
      if (opt == 'A') continue;
  
      chptr = strstr((char*)pOptionMapping, tstr);
      if (chptr == NULL) {
        throw std::runtime_error(std::string("Invalid command line option '-") 
            + static_cast<char>(opt) + "'");
      }
  
      chptr += 3;
      while (std::isspace(*chptr)) {
        chptr++;
      }
  
      if (!chptr || chptr[0] == '-') {// Option without format string will be ignored
        optfmt = " ";
      } else {
        optfmt = chptr;
        while (*chptr && !std::isspace(*chptr)) {
          chptr++;
        }
        if (!*chptr) {
          throw std::runtime_error("Fatal: Unexpected end of optionMap string");
        }
      }
      for (i = 0; !std::isspace(*optfmt); optfmt++) {
        while (std::isspace(*chptr)) chptr++;
        value = chptr;
        while (*chptr && !std::isspace(*chptr)) chptr++;
        assert(static_cast<unsigned int>(chptr-value+1) < sizeof(param));
        strncat(strcat(strcpy(param, pToolName), ":"), value, chptr-value);
        param[chptr-value+strlen(pToolName)+1] = '\0';
        switch (*optfmt) {
          case 'n': 
            value = strchr(param, '=');
            if (value) *value = '\0';
            InsertConfigParam(param,
                              value ? value + 1: "TRUE", opt);
            break;

          case 'l':
          case 'o':
          case 'r': 
            i++;
            if (!optarg && (optind+1==argc || IS_OPTION(argv[optind+1]))) {
              if (*optfmt == 'r' || *optfmt == 'l') {
                throw std::runtime_error(std::string("Argument ") 
                    + (ss<<i,ss).str() + " of option '-" 
                    + static_cast<char>(opt) + "' expected");
              }
              optfmt = "  "; // Stop reading option arguments
              break;
            }
            if (!optarg) optarg = argv[++optind];
            if (*optfmt == 'o') {
              option_must_follow = (bool) 1;
            }
            bptr = NULL;

            // For repeated use of option with 'l' (list) format, append
            // ',' and argument string to existing config parameter value.
            if (*optfmt == 'l' && OPTION_MARK(opt)) {
              bptr = strdup(GetStr(param, ""));
              if (bptr == NULL) throw std::runtime_error("Insufficient memory");
              bptr = (char*) realloc(bptr, strlen(bptr) + strlen(optarg) + 2);
              if (bptr == NULL) throw std::runtime_error("Insufficient memory");
              strcat(strcat(bptr, ","), optarg);
              optarg = bptr;
            }
            MARK_OPTION(opt);
            InsertConfigParam(param, optarg, opt);
            free(bptr);
            optarg = NULL;
            break;

          default : 
            throw std::runtime_error(std::string("Fatal: Invalid character '")
                + *optfmt + "' in optionMap after " + tstr);
        }
      }
      if (optarg) {
        throw std::runtime_error(std::string("Unexpected argument '")
            + optarg + "' after option '-" 
            + static_cast<char>(opt) + "'");
      }
    }
  
    for (i = optind; i < argc && !IS_OPTION(argv[i]); i++)
    {}
   
    if (i < argc) {
      throw std::runtime_error(std::string("No option expected after first non-option argument '")
          + argv[optind] + "'");
    }

    if (option_must_follow) {
      throw std::runtime_error(std::string("Option '-")
          + static_cast<char>(opt) 
          + "' with optional argument must not be the last option");
    }

    return optind;
  }
  

  //***************************************************************************
  //***************************************************************************
  int 
  UserInterface::
  GetFeatureParams(
    int *           derivOrder,
    int **          derivWinLens,
    int *           startFrmExt,
    int *           endFrmExt,
    char **         CMNPath,
    char **         CMNFile,
    const char **   CMNMask,
    char **         CVNPath,
    char **         CVNFile,
    const char **   CVNMask,
    const char **   CVGFile,
    const char *    pToolName,
    int             pseudoModeule)
  {
    const char *  str;
    int           targetKind;
    char *        chrptr;
    char          paramName[32];
    const char *  CMNDir;
    const char *  CVNDir;
    
    strcpy(paramName, pToolName);
    strcat(paramName, pseudoModeule == 1 ? "SPARM1:" :
                      pseudoModeule == 2 ? "SPARM2:" : "");
                      
    chrptr = paramName + strlen(paramName);
  
    strcpy(chrptr, "STARTFRMEXT");
    *startFrmExt = GetInt(paramName, 0);
    strcpy(chrptr, "ENDFRMEXT");
    *endFrmExt   = GetInt(paramName, 0);
  
    *CMNPath = *CVNPath = NULL;
    strcpy(chrptr, "CMEANDIR");
    CMNDir       = GetStr(paramName, NULL);
    strcpy(chrptr, "CMEANMASK");
    *CMNMask     = GetStr(paramName, NULL);

    if (*CMNMask != NULL) {
      *CMNPath = (char*) malloc((CMNDir ? strlen(CMNDir) : 0) + npercents(*CMNMask) + 2);
      if (*CMNPath == NULL) throw std::runtime_error("Insufficient memory");
      if (CMNDir != NULL) strcat(strcpy(*CMNPath, CMNDir), "/");
      *CMNFile = *CMNPath + strlen(*CMNPath);
    }
    strcpy(chrptr, "VARSCALEDIR");
    CVNDir      = GetStr(paramName, NULL);
    strcpy(chrptr, "VARSCALEMASK");
    *CVNMask     = GetStr(paramName, NULL);


    if (*CVNMask != NULL) {
      *CVNPath = (char*) malloc((CVNDir ? strlen(CVNDir) : 0) + npercents(*CVNMask) + 2);
      if (*CVNPath == NULL) throw std::runtime_error("Insufficient memory");
      if (CVNDir != NULL) strcat(strcpy(*CVNPath, CVNDir), "/");
      *CVNFile = *CVNPath + strlen(*CVNPath);
    }
    strcpy(chrptr, "VARSCALEFN");
    *CVGFile     = GetStr(paramName, NULL);
    strcpy(chrptr, "TARGETKIND");
    str = GetStr(paramName, "ANON");

    targetKind = FeatureRepository::ReadParmKind(str, false);

    if (targetKind == -1) {
      throw std::runtime_error(std::string("Invalid TARGETKIND = '")
          + str + "'");
    }
  
    strcpy(chrptr, "DERIVWINDOWS");
    if ((str = GetStr(paramName, NULL)) != NULL) {
      long lval;
      *derivOrder      = 0;
      *derivWinLens = NULL;
      
      if (NULL != str)
      {
        while ((str = strtok((char *) str, " \t_")) != NULL) 
        {
          lval = strtol(str, &chrptr, 0);
          if (!*str || *chrptr) {
            throw std::runtime_error("Integers separated by '_' expected for parameter DERIVWINDOWS");
          }
          *derivWinLens = (int *)realloc(*derivWinLens, ++*derivOrder*sizeof(int));
          if (*derivWinLens == NULL) throw std::runtime_error("Insufficient memory");
          (*derivWinLens)[*derivOrder-1] = lval;
          str = NULL;
        }
      }
      
      return targetKind;
    }
    *derivOrder = targetKind & PARAMKIND_T ? 3 :
                  targetKind & PARAMKIND_A ? 2 :
                  targetKind & PARAMKIND_D ? 1 : 0;
  
    if (*derivOrder || targetKind != PARAMKIND_ANON) {
    *derivWinLens = (int *) malloc(3 * sizeof(int));
      if (*derivWinLens == NULL) throw std::runtime_error("Insufficient memory");
  
      strcpy(chrptr, "DELTAWINDOW");
      (*derivWinLens)[0] = GetInt(paramName, 2);
      strcpy(chrptr, "ACCWINDOW");
      (*derivWinLens)[1] = GetInt(paramName, 2);
      strcpy(chrptr, "THIRDWINDOW");
      (*derivWinLens)[2] = GetInt(paramName, 2);
      return targetKind;
    }
    *derivWinLens = NULL;
    *derivOrder   = -1;
    return targetKind;
  }
  

  //***************************************************************************
  //***************************************************************************
  UserInterface::ValueRecord*
  UserInterface::
  GetParam(const char* pParamName)
  {
    MapType::iterator it;

    // this is done only for convenience. in the loop we will increase the 
    // pointer again
    pParamName--;

    // we iteratively try to find the param name in the map. if an attempt 
    // fails, we strip off all characters until the first ':' and we search 
    // again
    do {
      pParamName++;
      it = mMap.find(pParamName);
    } while ((it == mMap.end()) && (NULL != (pParamName = strchr(pParamName, ':'))));

    if (it == mMap.end()) {
      return NULL;
    }
    else {
      it->second.mRead = true;
      return &(it->second);
    }
  }
  

  //***************************************************************************
  //***************************************************************************
  const char * 
  UserInterface::
  GetStr(
    const char *    pParamName,
    const char *    default_value)
  {
    ValueRecord* p_val = GetParam(pParamName);

    if (NULL == p_val) {
      return default_value;
    }
    else {
      return p_val->mValue.c_str();
    }
  }
  

  //***************************************************************************
  //***************************************************************************
  long 
  UserInterface::
  GetInt(
    const char *pParamName,
    long default_value)
  {
    char *chrptr;
    ValueRecord* p_val = GetParam(pParamName);

    if (NULL == p_val) {
      return default_value;
    }
  
    const char *val = p_val->mValue.c_str();
    default_value = strtol(val, &chrptr, 0);
    if (!*val || *chrptr) {
      throw std::runtime_error(std::string("Integer number expected for ") 
          + pParamName + " but found '" + val + "'");
    }
    return default_value;
  }
  
  //***************************************************************************
  //***************************************************************************
  float 
  UserInterface::
  GetFlt(
    const char *      pParamName,
    float             default_value)
  {
    char *chrptr;
    ValueRecord* p_val = GetParam(pParamName);

    if (NULL == p_val) {
      return default_value;
    }
  
    const char *val = p_val->mValue.c_str();
    default_value = strtod(val, &chrptr);
    if (!*val || *chrptr) {
      throw std::runtime_error(std::string("Decimal number expected for ") 
          + pParamName + " but found '" + val + "'");
    }
    return default_value;
  }
  
  //***************************************************************************
  //***************************************************************************
  bool 
  UserInterface::
  GetBool(
    const char *    pParamName,
    bool            default_value)
  {
    ValueRecord* p_val = GetParam(pParamName);

    if (NULL == p_val) {
      return default_value;
    }
  
    const char* val = p_val->mValue.c_str();

    if (!strcasecmp(val, "TRUE") || !strcmp(val, "T")) return 1;
    if (strcasecmp(val, "FALSE") && strcmp(val, "F")) {
      throw std::runtime_error(std::string("TRUE or FALSE expected for ")
          + pParamName + " but found '" + val + "'");
    }
    return false;
  }
  
  //***************************************************************************
  //***************************************************************************
  // '...' are pairs: string and corresponding integer value , terminated by NULL
  int 
  UserInterface::
  GetEnum(
    const char *    pParamName,
    int             default_value, 
    ...)  
  {
    ValueRecord* p_val = GetParam(pParamName);

    if (NULL == p_val) {
      return default_value;
    }

    const char* val = p_val->mValue.c_str();
    char*       s;
    int i = 0, cnt = 0, l = 0;
    va_list ap;
  
    va_start(ap, default_value);
    while ((s = va_arg(ap, char *)) != NULL) {
      l += strlen(s) + 2;
      ++cnt;
      i = va_arg(ap, int);
      if (!strcmp(val, s)) break;
    }
    va_end(ap);

    if (s) {
      return i;
    }
  
    //To report error, create string listing all possible values
    s = (char*) malloc(l + 1);
    s[0] = '\0';
    va_start(ap, default_value);
    for (i = 0; i < cnt; i++) {
      strcat(s, va_arg(ap, char *));
      va_arg(ap, int);
      if (i < cnt - 2) strcat(s, ", ");
      else if (i == cnt - 2) strcat(s, " or ");
    }

    va_end(ap);

    throw std::runtime_error(std::string(s) + " expected for "
        + pParamName + " but found '" + val + "'");

    return 0;
  }
  

  //***************************************************************************
  //***************************************************************************
  void
  UserInterface::
  PrintConfig(std::ostream& rStream)
  {
    rStream << "Configuration Parameters[" << mMap.size() << "]\n";
    for (MapType::iterator it = mMap.begin(); it != mMap.end(); ++it) {
      rStream << (it->second.mRead ? "  " : "# ") 
        << std::setw(35) << std::left << it->first << " = "
        << std::setw(30) << std::left << it->second.mValue 
        << " # -" << it->second.mOption << std::endl;
    }
  }
  
  //***************************************************************************
  //***************************************************************************
  void 
  UserInterface::
  CheckCommandLineParamUse()
  {
    for (MapType::iterator it = mMap.begin(); it != mMap.end(); ++it) {
      if (!it->second.mRead && it->second.mOption != 'C') {
        Error("Unexpected command line parameter " + it->first);
      }
    }
  }

}
