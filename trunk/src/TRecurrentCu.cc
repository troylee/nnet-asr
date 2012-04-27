
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

#define SVN_DATE       "$Date: 2011-10-18 12:42:04 +0200 (Tue, 18 Oct 2011) $"
#define SVN_AUTHOR     "$Author: iveselyk $"
#define SVN_REVISION   "$Revision: 86 $"
#define SVN_ID         "$Id: TRecurrentCu.cc 86 2011-10-18 10:42:04Z iveselyk $"

#define MODULE_VERSION "1.0.0 "__TIME__" "__DATE__" "SVN_ID  




/*** TNetLib includes */
#include "Error.h"
#include "Timer.h"
#include "Features.h"
#include "Labels.h"
#include "Common.h"
#include "MlfStream.h"
#include "UserInterface.h"
#include "Timer.h"

/*** TNet includes */
#include "cuObjectiveFunction.h"
#include "cuNetwork.h"
#include "cuRecurrent.h"

/*** STL includes */
#include <iostream>
#include <sstream>
#include <numeric>




//////////////////////////////////////////////////////////////////////
// DEFINES
//

#define SNAME "TNET"

using namespace TNet;

void usage(const char* progname) 
{
  const char *tchrptr;
  if ((tchrptr = strrchr(progname, '\\')) != NULL) progname = tchrptr+1;
  if ((tchrptr = strrchr(progname, '/')) != NULL) progname = tchrptr+1;
  fprintf(stderr,
"\n%s version " MODULE_VERSION "\n"
"\nUSAGE: %s [options] DataFiles...\n\n"
"\n:TODO:\n\n"
" Option                                                     Default\n\n"
" -c         Enable crossvalidation                          off\n"
" -m file    Set label map of NN outputs                     \n"
" -n f       Set learning rate to f                          0.06\n"
" -o ext     Set target model ext                            None\n"
" -A         Print command line arguments                    Off\n" 
" -C cf      Set config file to cf                           Default\n"
" -D         Display configuration variables                 Off\n"
" -H mmf     Load NN macro file                              \n"
" -I mlf     Load master label file mlf                      \n"
" -L dir     Set input label (or net) dir                    Current\n"
" -M dir     Dir to write NN macro files                     Current\n"
" -O fn      Objective function [mse,xent]                   xent\n"
" -S file    Set script file                                 None\n"
" -T N       Set trace flags to N                            0\n" 
" -V         Print version information                       Off\n"
" -X ext     Set input label file ext                        lab\n"
"\n"
"BUNCHSIZE CACHESIZE CROSSVALIDATE FEATURETRANSFORM LEARNINGRATE LEARNRATEFACTORS MLFTRANSC MOMENTUM NATURALREADORDER OBJECTIVEFUNCTION OUTPUTLABELMAP PRINTCONFIG PRINTVERSION RANDOMIZE SCRIPT SEED SOURCEMLF SOURCEMMF SOURCETRANSCDIR SOURCETRANSCEXT TARGETMMF TARGETMODELDIR TARGETMODELEXT TRACE WEIGHTCOST\n"
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
    " -m r   OUTPUTLABELMAP" 
    " -n r   LEARNINGRATE" 
    " -D n   PRINTCONFIG=TRUE"
    " -H l   SOURCEMMF"
    " -I r   SOURCEMLF"
    " -L r   SOURCETRANSCDIR"
    " -S l   SCRIPT"
    " -T r   TRACE"
    " -V n   PRINTVERSION=TRUE"
    " -X r   SOURCETRANSCEXT";


  UserInterface        ui;
  FeatureRepository    feature_repo;
  LabelRepository      label_repo;
  CuNetwork            network;
  CuNetwork            transform_network;
  CuObjectiveFunction* p_obj_function = NULL;
  Timer                timer;
  Timer                timer_frontend;
  double               time_frontend = 0.0;

  const char*                       p_source_mmf_file;
  const char*                       p_input_transform;
  const char*                       p_targetmmf;
 
  const char*                       p_script;
  const char*                       p_output_label_map;

  BaseFloat                         learning_rate;
  const char*                       learning_rate_factors;
  BaseFloat                         momentum;
  BaseFloat                         weightcost;
  int                               bptt;
  CuObjectiveFunction::ObjFunType   obj_fun_id;

  const char*                       p_source_mlf_file;
  const char*                       p_src_lbl_dir;
  const char*                       p_src_lbl_ext;

  bool                              cross_validate;

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
  if (argc == 1) { usage(argv[0]); return 1; }
  int args_parsed = ui.ParseOptions(argc, argv, p_option_string, SNAME);


  // OPTION RETRIEVAL ........................................................
  // extract the feature parameters
  swap_features = !ui.GetBool(SNAME":NATURALREADORDER", TNet::IsBigEndian());
  
  target_kind = ui.GetFeatureParams(&deriv_order, &p_deriv_win_lenghts,
       &start_frm_ext, &end_frm_ext, &cmn_path, &cmn_file, &cmn_mask,
       &cvn_path, &cvn_file, &cvn_mask, &cvg_file, SNAME":", 0);


  // extract other parameters
  p_source_mmf_file   = ui.GetStr(SNAME":SOURCEMMF",     NULL);
  p_input_transform   = ui.GetStr(SNAME":FEATURETRANSFORM",  NULL);
  
  p_targetmmf         = ui.GetStr(SNAME":TARGETMMF",     NULL);

  p_script            = ui.GetStr(SNAME":SCRIPT",         NULL);
  p_output_label_map  = ui.GetStr(SNAME":OUTPUTLABELMAP", NULL);

  learning_rate       = ui.GetFlt(SNAME":LEARNINGRATE"  , 0.06f);
  learning_rate_factors = ui.GetStr(SNAME":LEARNRATEFACTORS", NULL);
  momentum            = ui.GetFlt(SNAME":MOMENTUM"      , 0.0);
  weightcost          = ui.GetFlt(SNAME":WEIGHTCOST"    , 0.0);
  bptt                = ui.GetInt(SNAME":BPTT"          , 4);

  obj_fun_id          = static_cast<CuObjectiveFunction::ObjFunType>(
                        ui.GetEnum(SNAME":OBJECTIVEFUNCTION", 
                                   CuObjectiveFunction::CROSS_ENTROPY, //< default
                                   "xent", CuObjectiveFunction::CROSS_ENTROPY,
                                   "mse", CuObjectiveFunction::MEAN_SQUARE_ERROR
                        ));



  p_source_mlf_file   = ui.GetStr(SNAME":SOURCEMLF",       NULL);
  p_src_lbl_dir       = ui.GetStr(SNAME":SOURCETRANSCDIR", NULL);
  p_src_lbl_ext       = ui.GetStr(SNAME":SOURCETRANSCEXT", "lab");

  cross_validate      = ui.GetBool(SNAME":CROSSVALIDATE",  false);

  trace               = ui.GetInt(SNAME":TRACE",               0);
  //if(trace&1) { 
    CuDevice::Instantiate().Verbose(true); 
  //}


  //throw away...
  ui.GetInt(SNAME":BUNCHSIZE", 256);
  ui.GetInt(SNAME":CACHESIZE", 12800);
  ui.GetBool(SNAME":RANDOMIZE", true);


  // process the parameters
  if(ui.GetBool(SNAME":PRINTCONFIG", false)) {
    std::cout << std::endl;
    ui.PrintConfig(std::cout);
    std::cout << std::endl;
  }
  if(ui.GetBool(SNAME":PRINTVERSION", false)) {
    std::cout << std::endl;
    std::cout << "======= "MODULE_VERSION" =======" << std::endl;
    std::cout << std::endl;
  }
  ui.CheckCommandLineParamUse();
  

  // the rest of the parameters are the feature files
  for (; args_parsed < argc; args_parsed++) {
    feature_repo.AddFile(argv[args_parsed]);
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


  // initialize the feature repository 
  feature_repo.Init(
    swap_features, start_frm_ext, end_frm_ext, target_kind,
    deriv_order, p_deriv_win_lenghts, 
    cmn_path, cmn_mask, cvn_path, cvn_mask, cvg_file
  );
  if(NULL != p_script) {
    feature_repo.AddFileList(p_script);
  } else {
    Warning("WARNING: The script file is missing [-S]");
  }

  // initialize the label repository
  if(NULL == p_source_mlf_file)
    Error("Source mlf file file is missing [-I]");
  if(NULL == p_output_label_map)
    Error("Output label map is missing [-m]");
  label_repo.Init(p_source_mlf_file, p_output_label_map, p_src_lbl_dir, p_src_lbl_ext);

  //get objective function instance
  p_obj_function = CuObjectiveFunction::Factory(obj_fun_id);

  //set the learnrate, etc
  network.SetLearnRate(learning_rate, learning_rate_factors);
  network.SetMomentum(momentum);
  network.SetWeightcost(weightcost);

  //set the BPTT order
  for(int i=0; i<network.Layers(); i++) {
    if(network.Layer(i).GetType() == CuComponent::RECURRENT) {
      dynamic_cast<CuRecurrent&>(network.Layer(i)).BpttOrder(bptt);
    }
  }
  
  
  //**********************************************************************
  //**********************************************************************
  // INITIALIZATION DONE .................................................
  //
  // Start training
  timer.Start();
  if(cross_validate) {
    std::cout << "===== TRecurrentCu CROSSVAL STARTED =====" << std::endl;
  } else {
    std::cout << "===== TRecurrentCu TRAINING STARTED =====" << std::endl;
  }

  feature_repo.Rewind();
  
  //**********************************************************************
  //**********************************************************************
  // MAIN LOOP
  //
  int frames = 0;
  Matrix<BaseFloat> targets_host;
  CuMatrix<BaseFloat> feats, output, targets, globerr;
  for(feature_repo.Rewind(); !feature_repo.EndOfList(); feature_repo.MoveNext()) {
    
    timer_frontend.Start();
      
    Matrix<BaseFloat> feats_host, globerr_host;
    CuMatrix<BaseFloat> feats_original;
    CuMatrix<BaseFloat> feats_expanded;

    //read feats, perfrom feature transform
    feature_repo.ReadFullMatrix(feats_host);
    feats_original.CopyFrom(feats_host);
    transform_network.Propagate(feats_original,feats_expanded);

    //trim the start/end context
    int rows = feats_expanded.Rows()-start_frm_ext-end_frm_ext;
    feats.Init(rows,feats_expanded.Cols());
    feats.CopyRows(rows,start_frm_ext,feats_expanded,0);

    timer_frontend.End(); time_frontend += timer_frontend.Val();

    //read the targets
    label_repo.GenDesiredMatrix(targets_host,feats.Rows(),
                                feature_repo.CurrentHeader().mSamplePeriod,
                                feature_repo.Current().Logical().c_str());
    targets.CopyFrom(targets_host);

    //reset the history context
    for(int i=0; i<network.Layers(); i++) {
      if(network.Layer(i).GetType() == CuComponent::RECURRENT) {
        dynamic_cast<CuRecurrent&>(network.Layer(i)).ClearHistory();
      }
    }

    CuMatrix<BaseFloat> input_row(1,feats.Cols());
    CuMatrix<BaseFloat> output_row(1,network.GetNOutputs());
    CuMatrix<BaseFloat> target_row(1,network.GetNOutputs());
    CuMatrix<BaseFloat> error_row(1,network.GetNOutputs());
    for(size_t frm=0; frm<feats.Rows(); frm++) {
      //select data rows
      input_row.CopyRows(1,frm,feats,0);
      target_row.CopyRows(1,frm,targets,0);

      //forward
      network.Propagate(input_row,output_row);
      
      //xetropy
      p_obj_function->Evaluate(output_row,target_row,error_row);

      if(!cross_validate) {
        //backward
        network.Backpropagate(error_row);
      }
    }

    frames += feats.Rows();
    std::cout << "." << std::flush; 
  }



  //**********************************************************************
  //**********************************************************************
  // TRAINING FINISHED .................................................
  //
  // Let's store the network, report the log

 
  if(cross_validate) {
    if(trace&1) TraceLog("Crossval finished");
  } else {
    if(trace&1) TraceLog("Training finished");
  }

  //write the network
  if(!cross_validate) {
    if (NULL != p_targetmmf) {
      if(trace&1) TraceLog(std::string("Writing network: ")+p_targetmmf);
      network.WriteNetwork(p_targetmmf);
    } else {
      Error("forgot to specify --TARGETMMF argument");
    }
  }
  
  timer.End();
  std::cout << std::endl;
  std::cout << "===== TRecurrentCu FINISHED ( " << timer.Val() << "s ) "
            << "[FPS:" << float(frames) / timer.Val() 
            << ",RT:" << 1.0f / (float(frames) / timer.Val() / 100.0f)
            << "] =====" << std::endl;

  //report objective function (accuracy, frame counts...)
  std::cout << "-- " << (cross_validate?"CV":"TR") << p_obj_function->Report();
  std::cout << "T-fe: " << time_frontend << std::endl;
  
  return  0; ///finish OK

} catch (std::exception& rExc) {
  std::cerr << "Exception thrown" << std::endl;
  std::cerr << rExc.what() << std::endl;
  return  1;
}

