
/***************************************************************************
 *   copyright            : (C) 2011 by Karel Vesely,UPGM,FIT,VUT,Brno     *
 *   email                : iveselyk@fit.vutbr.cz                          *
 ***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the APACHE License as published by the          *
 *   Apache Software Foundation; either version 2.0 of the License,        *
 *   or (at your option) any later version.                                *
 *                                                                         *
 ***************************************************************************/

#define SVN_DATE       "$Date: 2011-03-24 14:59:03 +0100 (Thu, 24 Mar 2011) $"
#define SVN_AUTHOR     "$Author: iveselyk $"
#define SVN_REVISION   "$Revision: 42 $"
#define SVN_ID         "$Id: TSegmenter.cc 42 2011-03-24 13:59:03Z iveselyk $"

#define MODULE_VERSION "1.0.0 "__TIME__" "__DATE__" "SVN_ID  




/*** TNetLib includes */
#include "Error.h"
#include "Timer.h"
#include "Features.h"
#include "Common.h"
#include "MlfStream.h"
#include "UserInterface.h"
#include "Timer.h"

/*** STL includes */
#include <iostream>
#include <sstream>
#include <numeric>

/*** Unix includes */
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>






//////////////////////////////////////////////////////////////////////
// DEFINES
//

#define SNAME "TSEGMNTER"

using namespace TNet;

void usage(const char* progname) 
{
  const char *tchrptr;
  if ((tchrptr = strrchr(progname, '\\')) != NULL) progname = tchrptr+1;
  if ((tchrptr = strrchr(progname, '/')) != NULL) progname = tchrptr+1;
  fprintf(stderr,
"\n%s version " MODULE_VERSION "\n"
"\nUSAGE: %s [options] DataFiles...\n\n"
" Option                                                     Default\n\n"
" -l dir     Set target directory for features               !REQ!\n"
//" -y ext     Set target feature ext                          fea_join\n"
" -A         Print command line arguments                    Off\n" 
" -C cf      Set config file to cf                           Default\n"
" -D         Display configuration variables                 Off\n"
" -S file    Set script file                                 None\n"
" -T N       Set trace flags to N                            0\n" 
" -V         Print version information                       Off\n"
"\n"
"NATURALREADORDER NOSUBDIRS OUTPUTSCRIPT PRINTCONFIG PRINTVERSION SCRIPT TARGETPARAMDIR "/*TARGETPARAMEXT*/" TRACE\n"
"\n"
"STARTFRMEXT ENDFRMEXT CMEANDIR CMEANMASK VARSCALEDIR VARSCALEMASK VARSCALEFN TARGETKIND DERIVWINDOWS DELTAWINDOW ACCWINDOW THIRDWINDOW\n"
"\n"
" %s is Copyright (C) 2010-2011 Karel Vesely\n"
" licensed under the APACHE License, version 2.0\n"
" Bug reports, feedback, etc, to: iveselyk@fit.vutbr.cz\n"
"\n", progname, progname, progname);
  exit(-1);
}


///////////////////////////////////////////////////////////////////////
// MAIN FUNCTION
//


int main(int argc, char *argv[]) try
{
  const char* p_option_string =
    " -l r   TARGETPARAMDIR"
//    " -y r   TARGETPARAMEXT"
    " -D n   PRINTCONFIG=TRUE"
    " -S l   SCRIPT"
    " -T r   TRACE"
    " -V n   PRINTVERSION=TRUE"
    ;


  UserInterface        ui;
  FeatureRepository    features;
  //InputDataProxy       data_proxy;
  //Network              network;
  //ObjectiveFunction*             p_obj_function = NULL;
  Timer                timer;

 
  const char*                       p_script;
  const char*                       p_tgt_param_dir;
//  const char*                       p_tgt_param_ext;
  const char*                       p_output_script;
  int                               trace;
  bool                              create_subdirs;

  // variables for feature repository
  bool                              swap_features;
  int                               target_kind;
  int                               deriv_order;
  int*                              p_deriv_win_lenghts;
  int                               start_frm_ext;
  int                               end_frm_ext;
        char*                       cmn_path;
        char*                       cmn_file;
  const char*                       cmn_mask;
        char*                       cvn_path;
        char*                       cvn_file;
  const char*                       cvn_mask;
  const char*                       cvg_file;

 
  // OPTION PARSING ..........................................................
  // use the STK option parsing
  if (argc == 1) { usage(argv[0]); return 1; }
  int args_parsed = ui.ParseOptions(argc, argv, p_option_string, SNAME);


  // OPTION RETRIEVAL ........................................................
  // extract the feature parameters
  swap_features = !ui.GetBool(SNAME":NATURALREADORDER", TNet::IsBigEndian());
  
  target_kind = ui.GetFeatureParams(&deriv_order, &p_deriv_win_lenghts,
       &start_frm_ext, &end_frm_ext, &cmn_path, &cmn_file, &cmn_mask,
       &cvn_path, &cvn_file, &cvn_mask, &cvg_file, SNAME":", 0);


  // extract other parameters
  p_script            = ui.GetStr(SNAME":SCRIPT",         NULL);
  p_tgt_param_dir     = ui.GetStr(SNAME":TARGETPARAMDIR", NULL);
//  p_tgt_param_ext     = ui.GetStr(SNAME":TARGETPARAMEXT", NULL);
  p_output_script     = ui.GetStr(SNAME":OUTPUTSCRIPT",   NULL);
  create_subdirs      = !ui.GetBool(SNAME":NOSUBDIRS", false);
  trace               = ui.GetInt(SNAME":TRACE",          00);


  // process the parameters
  if(ui.GetBool(SNAME":PRINTCONFIG", false)) {
    std::cout << std::endl;
    ui.PrintConfig(std::cout);
    std::cout << std::endl;
  }
  if(ui.GetBool(SNAME":PRINTVERSION", false)) {
    std::cout << std::endl;
    std::cout << "======= TNET v"MODULE_VERSION" xvesel39 =======" << std::endl;
    std::cout << std::endl;
  }
  ui.CheckCommandLineParamUse();
  

  // the rest of the parameters are the feature files
  for (; args_parsed < argc; args_parsed++) {
    features.AddFile(argv[args_parsed]);
  }

  //**************************************************************************
  //**************************************************************************
  // OPTION PARSING DONE .....................................................


  //initialize FeatureRepository
  features.AddFileList(p_script);
  
  features.Init(
    swap_features, start_frm_ext, end_frm_ext, target_kind,
    deriv_order, p_deriv_win_lenghts, 
    cmn_path, cmn_mask, cvn_path, cvn_mask, cvg_file
  );

  //start timer
  timer.Start();

  std::cout << "[Segmentation started]" << std::endl;

  //segment the features
  size_t cnt = 0;
  size_t step = features.QueueSize() / 100;
  if(step == 0) step = 1;

  //open output script file
  std::ofstream out_scp;
  if(NULL == p_output_script) Error("OUTPUTSCRIPT parameter needed");
  out_scp.open(p_output_script);
  if(!out_scp.good()) Error(std::string("Cannot open output script file")+p_output_script);

  //store short segments of the data
  Matrix<BaseFloat> matrix;
  std::string file_out;

  features.Rewind();
  for( ; !features.EndOfList(); features.MoveNext(), cnt++) {
    //read the features
    features.ReadFullMatrix(matrix);

    //build the output feature filename
    file_out = "";
    if(NULL != p_tgt_param_dir) {
      (file_out += p_tgt_param_dir) += "/";
    }
    
    //create directory structure
    if(create_subdirs) {
      char subd[64];
      sprintf(subd,"%06d/",cnt/1000);
      file_out += subd;
      //create dir
      if(access(file_out.c_str(), R_OK|W_OK|X_OK)) {
        if(mkdir(file_out.c_str(),0770)) {
          Error(std::string("Cannot create directory:")+file_out);
        }
      }
    }

    //append logical filename
    file_out += features.Current().Logical();

    //get the targetkind and source_rate 
    if(target_kind == PARAMKIND_ANON) {
      target_kind = features.CurrentHeader().mSampleKind;
    }
    int source_rate = features.CurrentHeader().mSamplePeriod;
    //write the output feature
    features.WriteFeatureMatrix(matrix, file_out, target_kind, source_rate);
    //write the output scriptfile record
    out_scp << file_out << "[" << start_frm_ext << "," << matrix.Rows()-end_frm_ext-1 << "]\n";
    out_scp << std::flush;

    if((cnt % step) == 0) std::cout << 100 * cnt / features.QueueSize() << "%, " << std::flush;
  }

  //close output script file
  out_scp.close();

  timer.End();
  std::cout << "\n[Segmentation finished, elapsed time:( " << timer.Val() <<"s )]" << std::endl;


  return  0; ///finish OK

} catch (std::exception& rExc) {
  std::cerr << "Exception thrown" << std::endl;
  std::cerr << rExc.what() << std::endl;
  return  1;
}
