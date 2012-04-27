#ifndef TNet_UserInterface_h
#define TNet_UserInterface_h

#include <iostream>
#include <cstdlib>
#include <string>
#include <map>

namespace TNet 
{
  /** **************************************************************************
   ** **************************************************************************
   */
  class UserInterface 
  {
  public:
    struct ValueRecord {
      std::string   mValue;
      char          mOption;
      bool          mRead;
    };


    void InsertConfigParam(
      const char *param_name,
      const char *value,
      int optionChar);
    

    void 
    ReadConfig(const char *pFileName);


    void 
    CheckCommandLineParamUse();
    

    /** 
     * @brief Retreives the content of a parameter
     * @param pParamName Name of the parameter to look for
     * @return Returns the pointer to the ValueRecord structure if success,
     *         otherwise return NULL
     *
     *  We iteratively try to find the param name in the map. If an attempt 
     *  fails, we strip off all characters until the first occurance of ':' 
     *  and we search again
     */
    ValueRecord*
    GetParam(const char* pParamName);


    /** 
     * @brief Returns the parameter's value as string
     * 
     * @param param_name Parameter name
     * @param default_value Value, which is returned in case the parameter 
     * was not found
     * 
     * @return Pointer to the begining of the string if success, default_value
     * otherwise
     */
    const char* 
    GetStr( const char *param_name, const char *default_value);
    

    /** 
     * @brief Returns the parameter's value as int
     * 
     * @param param_name Parameter name
     * @param default_value Value, which is returned in case the parameter 
     * was not found
     * 
     * @return Returns the integer value if success, default_value
     * otherwise
     */
    long 
    GetInt( const char *param_name, long default_value);
    

    /** 
     * @brief Returns the parameter's value as float
     * 
     * @param param_name Parameter name
     * @param default_value Value, which is returned in case the parameter 
     * was not found
     * 
     * @return Returns the float value if success, default_value
     * otherwise
     */
    float 
    GetFlt( const char *param_name, float default_value);
    

    /** 
     * @brief Returns the parameter's value as bool
     * 
     * @param param_name Parameter name
     * @param default_value Value, which is returned in case the parameter 
     * was not found
     * 
     * @return Returns the bool value if success, default_value
     * otherwise
     *
     * Note that true is returned if the value is 'TRUE' or 'T', false is
     * returned if the value is 'FALSE' or 'F'. Otherwise exception is thrown
     */
    bool 
    GetBool(const char *param_name, bool default_value);
    

    /** 
     * @brief Returns the parameter's value as enum integer
     * 
     * @param param_name Parameter name
     * @param default_value Value, which is returned in case the parameter 
     * was not found
     * 
     * @return Returns the index value if success, default_value
     * otherwise
     *
     * Variable arguments specify the possible values of this parameter. If the
     * value does not match any of these, exception is thrown.
     */
    int 
    GetEnum( const char *param_name, int default_value, ...);
    

    int GetFeatureParams(
        int *derivOrder,
        int **derivWinLens,
        int *startFrmExt,
        int *endFrmExt,
        char **CMNPath,
        char **CMNFile,
        const char **CMNMask,
        char **CVNPath,
        char **CVNFile,
        const char **CVNMask,
        const char **CVGFile,
        const char *toolName,
        int pseudoModeule);
    

    int ParseOptions(
        int             argc,
        char*           argv[],
        const char*     optionMapping,
        const char*     toolName);


    /** 
     * @brief Send the defined paramaters to a stream
     * 
     * @param rStream stream to use
     */
    void
    PrintConfig(std::ostream& rStream);

  public:
    typedef std::map<std::string, ValueRecord> MapType;
    MapType             mMap;
  };
}

#endif
  
