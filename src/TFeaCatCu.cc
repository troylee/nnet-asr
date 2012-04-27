
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


#define SVN_DATE       "$Date: 2012-01-27 16:33:21 +0100 (Fri, 27 Jan 2012) $"
#define SVN_AUTHOR     "$Author: iveselyk $"
#define SVN_REVISION   "$Revision: 98 $"
#define SVN_ID         "$Id: TFeaCatCu.cc 98 2012-01-27 15:33:21Z iveselyk $"

#define MODULE_VERSION "1.0.0 "__TIME__" "__DATE__" "SVN_ID  





#include "Error.h"
#include "Timer.h"
#include "Features.h"
#include "Common.h"
#include "UserInterface.h"

#include "cuNetwork.h"

#include <iostream>
#include <sstream>



//////////////////////////////////////////////////////////////////////
// DEFINES
//

#define SNAME "TFEACAT"

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
" -l dir     Set target directory for features               Current\n"
" -y ext     Set target feature ext                          fea\n"
" -A         Print command line arguments                    Off\n" 
" -C cf      Set config file to cf                           Default\n"
" -D         Display configuration variables                 Off\n"
" -H mmf     Load NN macro file                              \n"  
" -S file    Set script file                                 None\n"
" -T N       Set trace flags to N                            0\n"
" -V         Print version information                       Off\n"
"\n"
"FEATURETRANSFORM GMMBYPASS LOGPOSTERIOR NATURALREADORDER PRINTCONFIG PRINTVERSION SCRIPT SOURCEMMF TARGETPARAMDIR TARGETPARAMEXT TRACE\n"
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
    " -y r   TARGETPARAMEXT" 
    " -D n   PRINTCONFIG=TRUE"
    " -H l   SOURCEMMF"
    " -S l   SCRIPT"
    " -T r   TRACE"
    " -V n   PRINTVERSION=TRUE";

  if(argc == 1) { usage(argv[0]); }

  UserInterface        ui;
  FeatureRepository    feature_repo;
  CuNetwork            transform_network;
  CuNetwork            network;
  Timer                tim;

 
  const char*                       p_script;
        char                        p_target_fea[4096];
  const char*                       p_target_fea_dir;
  const char*                       p_target_fea_ext;

  const char*                       p_source_mmf_file;
  const char*                       p_input_transform;

  bool                              gmm_bypass;
  bool                              log_posterior;
  int                               trace;

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
  int ii = ui.ParseOptions(argc, argv, p_option_string, SNAME);


  // OPTION RETRIEVAL ........................................................
  // extract the feature parameters
  swap_features = !ui.GetBool(SNAME":NATURALREADORDER", TNet::IsBigEndian());
  
  target_kind = ui.GetFeatureParams(&deriv_order, &p_deriv_win_lenghts,
       &start_frm_ext, &end_frm_ext, &cmn_path, &cmn_file, &cmn_mask,
       &cvn_path, &cvn_file, &cvn_mask, &cvg_file, SNAME":", 0);


  // extract other parameters
  p_source_mmf_file   = ui.GetStr(SNAME":SOURCEMMF",     NULL);
  p_input_transform   = ui.GetStr(SNAME":FEATURETRANSFORM",  NULL);

  p_script            = ui.GetStr(SNAME":SCRIPT",         NULL);
  p_target_fea_dir    = ui.GetStr(SNAME":TARGETPARAMDIR", NULL);
  p_target_fea_ext    = ui.GetStr(SNAME":TARGETPARAMEXT", "fea");
 
  gmm_bypass          = ui.GetBool(SNAME":GMMBYPASS",     false);
  log_posterior       = ui.GetBool(SNAME":LOGPOSTERIOR",  false);
   
  trace               = ui.GetInt(SNAME":TRACE",          00);
  if(trace&1) { CuDevice::Instantiate().Verbose(true); }

  
  // process the parameters
  if(ui.GetBool(SNAME":PRINTVERSION", false)) {
    std::cout << "Version: "MODULE_VERSION"" << std::endl;
  }
  if(ui.GetBool(SNAME":PRINTCONFIG", false)) {
    std::cout << std::endl;
    ui.PrintConfig(std::cout);
    std::cout << std::endl;
  }
  ui.CheckCommandLineParamUse();
  

  // the rest of the parameters are the feature files
  for (; ii < argc; ii++) {
    feature_repo.AddFile(argv[ii]);
  }

  //**************************************************************************
  //**************************************************************************
  // OPTION PARSING DONE .....................................................

  //read the input transform network
  if(NULL != p_input_transform) { 
    if(trace&1) TraceLog(std::string("Reading input transform network: ")+p_input_transform);
    transform_network.ReadNetwork(p_input_transform);
  }

  //read the neural network
  if(NULL != p_source_mmf_file) { 
    if(trace&1) TraceLog(std::string("Reading network: ")+p_source_mmf_file);
    network.ReadNetwork(p_source_mmf_file);
  } else {
    Error("Source MMF must be specified [-H]");
  }
  
  //initialize the FeatureRepository
  feature_repo.Init(
    swap_features, start_frm_ext, end_frm_ext, target_kind,
    deriv_order, p_deriv_win_lenghts, 
    cmn_path, cmn_mask, cvn_path, cvn_mask, cvg_file
  );
  if(NULL != p_script) {
    feature_repo.AddFileList(p_script);
  } 
  if(feature_repo.QueueSize() <= 0) {
    KALDI_ERR << "No input features specified,\n"
              << " try [-S SCP] or positional argument";
  }
  

  //**************************************************************************
  //**************************************************************************
  // MAIN LOOP ...............................................................

  //progress
  size_t cnt = 0;
  size_t step = feature_repo.QueueSize() / 100;
  if(step == 0) step = 1;
  tim.Start();

  Matrix<BaseFloat> feats_in, feats_out;
  CuMatrix<BaseFloat> feats_in_cu, feats_transf_cu, feats_out_cu;
  //process all the feature files
  for(feature_repo.Rewind(); !feature_repo.EndOfList(); feature_repo.MoveNext()) {
    //read file
    feature_repo.ReadFullMatrix(feats_in);
    feats_in_cu.CopyFrom(feats_in);

    //apply input transform (even empty)
    transform_network.Propagate(feats_in_cu,feats_transf_cu);
    
    //propagate through the network
    network.Propagate(feats_transf_cu,feats_out_cu);
   
    //trim the start/end context
    int rows = feats_out_cu.Rows()-start_frm_ext-end_frm_ext;
    CuMatrix<BaseFloat> feats_trim_cu(rows,feats_out_cu.Cols());
    feats_trim_cu.CopyRows(rows,start_frm_ext,feats_out_cu,0);

    feats_trim_cu.CopyTo(feats_out);
    
    //GMM bypass for HVite using posteriors as features
    if(gmm_bypass) {
      for(size_t i=0; i<feats_out.Rows(); i++) {
        for(size_t j=0; j<feats_out.Cols(); j++) {
          feats_out(i,j) = static_cast<BaseFloat>(sqrt(-2.0*log(feats_out(i,j))));
        }
      }
    }
  
    //Convert posteriors to logdomain
    if(log_posterior) {
      for(size_t i=0; i<feats_out.Rows(); i++) {
        for(size_t j=0; j<feats_out.Cols(); j++) {
          feats_out(i,j) = static_cast<BaseFloat>(log(feats_out(i,j)));
        }
      }
    }

    
    //save output
    MakeHtkFileName(p_target_fea, feature_repo.Current().Logical().c_str(), p_target_fea_dir, p_target_fea_ext);
    int sample_period = feature_repo.CurrentHeader().mSamplePeriod;
    feature_repo.WriteFeatureMatrix(feats_out,p_target_fea,PARAMKIND_USER,sample_period);

    if(trace&1) {
      if((cnt++ % step) == 0) std::cout << 100 * cnt / feature_repo.QueueSize() << "%, " << std::flush;
    }
  }
    
  //finish
  if(trace&1) {
    tim.End();
    std::cout << "TFeaCat finished: " << tim.Val() << "s" << std::endl;
  }
  return 0;

} catch (std::exception& rExc) {
  std::cerr << "Exception thrown" << std::endl;
  std::cerr << rExc.what() << std::endl;
  return 1;
}
