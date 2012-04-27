
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

#define SVN_DATE       "$Date: 2012-03-23 14:22:49 +0100 (Fri, 23 Mar 2012) $"
#define SVN_AUTHOR     "$Author: iveselyk $"
#define SVN_REVISION   "$Revision: 110 $"
#define SVN_ID         "$Id: TJoiner.cc 110 2012-03-23 13:22:49Z iveselyk $"

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
#include <limits>

/*** Unix includes */
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>






//////////////////////////////////////////////////////////////////////
// DEFINES
//

#define SNAME "TJOINER"

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
" -y ext     Set target feature ext                          fea_join\n"
" -A         Print command line arguments                    Off\n" 
" -C cf      Set config file to cf                           Default\n"
" -D         Display configuration variables                 Off\n"
" -S file    Set script file                                 None\n"
" -T N       Set trace flags to N                            0\n" 
" -V         Print version information                       Off\n"
"\n"
"NATURALREADORDER OUTPUTSCRIPT PRINTCONFIG PRINTVERSION SCRIPT TARGETPARAMDIR TARGETPARAMEXT TARGETSIZE TRACE\n"
"\n"
"STARTFRMEXT ENDFRMEXT CMEANDIR CMEANMASK VARSCALEDIR VARSCALEMASK VARSCALEFN TARGETKIND DERIVWINDOWS DELTAWINDOW ACCWINDOW THIRDWINDOW\n"
"\n"
" %s is Copyright (C) 2010-2011 Karel Vesely\n"
" licensed under the APACHE License, version 2.0\n"
" Bug reports, feedback, etc, to: iveselyk@fit.vutbr.cz\n"
"\n", progname, progname, progname);
  exit(-1);
}



inline std::string int2str(int i) {
  char buf[64];
  sprintf(buf,"%06d",i);
  return buf;
}




///////////////////////////////////////////////////////////////////////
// MAIN FUNCTION
//


int main(int argc, char *argv[]) try
{
  const char* p_option_string =
    " -l r   TARGETPARAMDIR"
    " -y r   TARGETPARAMEXT"
    " -D n   PRINTCONFIG=TRUE"
    " -S l   SCRIPT"
    " -T r   TRACE"
    " -V n   PRINTVERSION=TRUE"
    ;


  UserInterface        ui;
  FeatureRepository    features;
  Timer                timer;

 
  const char*                       p_script;
  const char*                       p_tgt_param_dir;
  const char*                       p_tgt_param_ext;
  const char*                       p_output_script;
  int                               trace;
  int                               target_size;
  bool                              dir_strip;

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
  p_tgt_param_dir     = ui.GetStr(SNAME":TARGETPARAMDIR",      NULL);
  p_tgt_param_ext     = ui.GetStr(SNAME":TARGETPARAMEXT","fea_join");
  p_output_script     = ui.GetStr(SNAME":OUTPUTSCRIPT",   NULL);
  trace               = ui.GetInt(SNAME":TRACE",          00);
  target_size         = ui.GetInt(SNAME":TARGETSIZE",   20000);
  dir_strip           = ui.GetBool(SNAME":DIRSTRIP", true);

  // process the parameters
  if(ui.GetBool(SNAME":PRINTCONFIG", false)) {
    std::cout << std::endl;
    ui.PrintConfig(std::cout);
    std::cout << std::endl;
  }
  if(ui.GetBool(SNAME":PRINTVERSION", false)) {
    std::cout << std::endl;
    std::cout << "Version: "MODULE_VERSION"\n";
    std::cout << std::endl;
  }
  ui.CheckCommandLineParamUse();
  

  // the rest of the parameters are the feature files
  for (; args_parsed < argc; args_parsed++) {
    features.AddFile(argv[args_parsed]);
  }



  if(NULL == p_tgt_param_dir) {
    Error("OUTPUTDIR must be specified");
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

  std::cout << "[Feature joining started]" << std::endl;

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
  Matrix<BaseFloat> mat_in, mat_buffer, mat_out;
  Vector<BaseFloat> vec_sep;
  int pos_buf = 0;
  int dim = -1;
  
  int file_out_ctr = 1;
  std::string file_out;
  file_out = std::string(p_tgt_param_dir) + "/" + int2str(file_out_ctr) + "." + p_tgt_param_ext;

  features.Rewind();
  for( ; !features.EndOfList(); features.MoveNext(), cnt++) {
    //read the features
    features.ReadFullMatrix(mat_in);

    //skip invalid segments
    bool skip = false;
    for(size_t r=0; r<mat_in.Rows(); r++) {
      for(size_t c=0; c<mat_in.Cols(); c++) {
        if(isnan(mat_in(r,c)) || isinf(mat_in(r,c))) {
          skip = true;
        }
      }
    }
    if(skip) {
      Warning(std::string("Skipping:")+features.Current().Logical()+"\nIt contains nan or inf!!!");
      continue;
    }

    //lazy buffer init
    if(mat_buffer.Rows() == 0) {
      dim = mat_in.Cols();
      //init buffer
      mat_buffer.Init(target_size,dim);
      //set the separator frame to nan
      vec_sep.Init(dim);
      vec_sep.Set(std::numeric_limits<BaseFloat>::quiet_NaN());
    }

    if(pos_buf+1+mat_in.Rows() >= (unsigned)target_size) {
      mat_out.Init(pos_buf+mat_in.Rows(),dim);
      //copy buffer
      if(pos_buf > 0) {
        memcpy(mat_out.pData(),mat_buffer.pData(),pos_buf*mat_buffer.Stride()*sizeof(BaseFloat));
      }
      //copy matrix
      memcpy(mat_out.pRowData(pos_buf),mat_in.pData(),mat_in.MSize());
      //strip directory from logical filename
      std::string name_logical(features.Current().Logical());
      size_t str_pos;
      if(dir_strip && (str_pos = name_logical.rfind("/")) != std::string::npos) {
        name_logical.erase(0,str_pos+1);
      }
      //add scriptfile record
      out_scp << name_logical << "=" << file_out << "[" << pos_buf+start_frm_ext << "," << pos_buf+mat_in.Rows()-end_frm_ext-1 << "]\n";

      //save the file
      //get the targetkind and source_rate 
      if(target_kind == PARAMKIND_ANON) {
        target_kind = features.CurrentHeader().mSampleKind;
      }
      int source_rate = features.CurrentHeader().mSamplePeriod;
      //write the output feature
      features.WriteFeatureMatrix(mat_out, file_out, target_kind, source_rate);
      //get next filename
      file_out_ctr++;
      file_out = std::string(p_tgt_param_dir) + "/" + int2str(file_out_ctr) + "." + p_tgt_param_ext;

      //set the buffer empty
      pos_buf = 0;
      continue;
    }

    //strip directory from logical filename
    std::string name_logical(features.Current().Logical());
    size_t str_pos;
    if(dir_strip && (str_pos = name_logical.rfind("/")) != std::string::npos) {
      name_logical.erase(0,str_pos+1);
    }
    //add scriptfile record
    out_scp << name_logical << "=" << file_out << "[" << pos_buf+start_frm_ext << "," << pos_buf+mat_in.Rows()-end_frm_ext-1 << "]\n";

    //add mat_in to cache, add separator
    memcpy(mat_buffer.pRowData(pos_buf),mat_in.pData(),mat_in.MSize());
    pos_buf += mat_in.Rows();
    mat_buffer[pos_buf].Copy(vec_sep);
    pos_buf++;

    if((cnt % step) == 0) std::cout << 100 * cnt / features.QueueSize() << "%, " << std::flush;
  }

  //store the content of the buffer
  if(pos_buf > 0) {
    mat_out.Init(pos_buf-1,dim); //don't store separator! => -1
    memcpy(mat_out.pData(),mat_buffer.pData(),mat_out.MSize());
    //save the file
    //get the targetkind and source_rate 
    if(target_kind == PARAMKIND_ANON) {
      target_kind = features.CurrentHeader().mSampleKind;
    }
    int source_rate = features.CurrentHeader().mSamplePeriod;
    //write the output feature
    features.WriteFeatureMatrix(mat_out, file_out, target_kind, source_rate);
;
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

