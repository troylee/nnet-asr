
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

#define SVN_DATE       "$Date: 2011-12-08 11:59:03 +0100 (Thu, 08 Dec 2011) $"
#define SVN_AUTHOR     "$Author: iveselyk $"
#define SVN_REVISION   "$Revision: 94 $"
#define SVN_ID         "$Id: TRbmCu.cc 94 2011-12-08 10:59:03Z iveselyk $"

#define MODULE_VERSION "1.0.0 "__TIME__" "__DATE__" "SVN_ID  





/*** TNetLib includes */
#include "Error.h"
#include "Timer.h"
#include "Features.h"
#include "Common.h"
#include "UserInterface.h"
#include "Timer.h"

/*** TNet includes */
#include "cuNetwork.h"
#include "cuRbm.h"
#include "cuCache.h"
#include "cuObjectiveFunction.h"
#include "curand.h"

/*** STL includes */
#include <iostream>
#include <sstream>
#include <numeric>




//////////////////////////////////////////////////////////////////////
// DEFINES
//

#define SNAME "TRBM"

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
" -n f       Set learning rate to f                          0.06\n"
" -A         Print command line arguments                    Off\n" 
" -C cf      Set config file to cf                           Default\n"
" -D         Display configuration variables                 Off\n"
" -H mmf     Load NN macro file                              \n"
" -S file    Set script file                                 None\n"
" -T N       Set trace flags to N                            0\n" 
" -V         Print version information                       Off\n"
"\n"
"FEATURETRANSFORM LEARNINGRATE MOMENTUM NATURALREADORDER PRINTCONFIG PRINTVERSION SCRIPT SOURCEMMF TARGETMMF TRACE WEIGHTCOST\n"
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
    " -n r   LEARNINGRATE"
    " -D n   PRINTCONFIG=TRUE"
    " -H l   SOURCEMMF"
    " -S l   SCRIPT"
    " -T r   TRACE"
    " -V n   PRINTVERSION=TRUE"
    ;


  UserInterface        ui;
  FeatureRepository    feature_repo;
  CuNetwork            network;
  CuNetwork            transform_network;
  CuMeanSquareError    mse;
  Timer                timer;
  Timer                timer_frontend;
  double               time_frontend = 0.0;

 
  const char*                       p_script;
  BaseFloat                         learning_rate;
  BaseFloat                         momentum;
  BaseFloat                         weightcost;

  const char*                       p_source_mmf_file;
  const char*                       p_input_transform;

  const char*                       p_targetmmf; 

  int                               bunch_size;
  int                               cache_size;
  bool                              randomize;
  long int                          seed;
  
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
  learning_rate       = ui.GetFlt(SNAME":LEARNINGRATE"  , 0.10f);
  momentum            = ui.GetFlt(SNAME":MOMENTUM"      , 0.50f);
  weightcost          = ui.GetFlt(SNAME":WEIGHTCOST"    , 0.0002f);


  bunch_size          = ui.GetInt(SNAME":BUNCHSIZE", 256);
  cache_size          = ui.GetInt(SNAME":CACHESIZE", 12800);
  randomize           = ui.GetBool(SNAME":RANDOMIZE", true);

  //cannot get long int
  seed                = ui.GetInt(SNAME":SEED", 0);

  trace               = ui.GetInt(SNAME":TRACE", 0);
  if(trace&4) { CuDevice::Instantiate().Verbose(true); }




  // process the parameters
  if(ui.GetBool(SNAME":PRINTCONFIG", false)) {
    std::cout << std::endl;
    ui.PrintConfig(std::cout);
    std::cout << std::endl;
  }
  if(ui.GetBool(SNAME":PRINTVERSION", false)) {
    std::cout << std::endl;
    std::cout << "======= TRbmCu v"MODULE_VERSION" xvesel39 =======" << std::endl;
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
  //extract the RBM from the network
  if(network.Layers() != 1) { 
    Error(std::string("Number of layers must be 1")+p_source_mmf_file); 
  }
  if(network.Layer(0).GetType() != CuComponent::RBM && network.Layer(0).GetType() != CuComponent::RBM_SPARSE) {
    Error(std::string("Layer must be RBM")+p_source_mmf_file);
  }
  CuRbmBase& rbm = dynamic_cast<CuRbmBase&>(network.Layer(0));

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
  feature_repo.Trace(trace);

  //set the learnrate, momentum, weightcost
  rbm.LearnRate(learning_rate);
  rbm.Momentum(momentum);
  rbm.Weightcost(weightcost);

  //seed the random number generator
  if(seed == 0) {
    struct timeval tv;
    if (gettimeofday(&tv, 0) == -1) {
      assert(0 && "gettimeofday does not work.");
      exit(-1);
    }
    seed = (int)(tv.tv_sec) + (int)tv.tv_usec;
  }
  srand48(seed);

  //initialize the matrix random number generator
  CuRand<BaseFloat> cu_rand(bunch_size,rbm.GetNOutputs());


  
  //**********************************************************************
  //**********************************************************************
  // INITIALIZATION DONE .................................................
  //
  // Start training
  timer.Start();
  std::cout << "===== TRbmCu TRAINING STARTED =====" << std::endl;
  std::cout << "learning rate: " << learning_rate 
            << " momentum: " << momentum 
            << " weightcost: " << weightcost
            << std::endl;
  std::cout << "Using seed: " << seed << "\n";


  CuCache cache;
  cache.Init(cache_size,bunch_size);
  cache.Trace(trace);
  feature_repo.Rewind();
  
  //**********************************************************************
  //**********************************************************************
  // MAIN LOOP
  //
  CuMatrix<BaseFloat> pos_vis, pos_hid, neg_vis, neg_hid;
  CuMatrix<BaseFloat> dummy_labs, dummy_err;
  while(!feature_repo.EndOfList()) {
    timer_frontend.Start();
    //fill cache
    while(!cache.Full() && !feature_repo.EndOfList()) {
      Matrix<BaseFloat> feats_host;
      CuMatrix<BaseFloat> feats_original;
      CuMatrix<BaseFloat> feats_expanded;

      //read feats, perfrom feature transform
      feature_repo.ReadFullMatrix(feats_host);
      feats_original.CopyFrom(feats_host);
      transform_network.Propagate(feats_original,feats_expanded);

      //trim the start/end context
      int rows = feats_expanded.Rows()-start_frm_ext-end_frm_ext;
      CuMatrix<BaseFloat> feats_trim(rows,feats_expanded.Cols());
      feats_trim.CopyRows(rows,start_frm_ext,feats_expanded,0);

      //fake the labels!!!
      CuMatrix<BaseFloat> labs_cu(feats_trim.Rows(),1);
      
      //add to cache
      cache.AddData(feats_trim,labs_cu);

      feature_repo.MoveNext();
    }
    timer_frontend.End(); time_frontend += timer_frontend.Val();
   
    if(randomize) { 
      //randomize the cache
      cache.Randomize();
    }

    while(!cache.Empty()) {
      //get training data
      cache.GetBunch(pos_vis,dummy_labs);

      //forward pass
      rbm.Propagate(pos_vis,pos_hid);

      //change the hidden values so we can generate negative example
      if(rbm.HidType() == CuRbmBase::BERNOULLI) {
        cu_rand.BinarizeProbs(pos_hid,neg_hid);
      } else {
        neg_hid.CopyFrom(pos_hid);
        cu_rand.AddGaussNoise(neg_hid);
      }

      //reconstruct pass
      rbm.Reconstruct(neg_hid,neg_vis);

      //forward pass
      rbm.Propagate(neg_vis, neg_hid);

      //update the weioghts
      rbm.RbmUpdate(pos_vis, pos_hid, neg_vis, neg_hid);

      //evalueate mean square error
      mse.Evaluate(neg_vis,pos_vis,dummy_err);

      if(trace&2) std::cout << "." << std::flush;
    }
    //check the NaN/inf
    pos_hid.CheckData();
  }



  //**********************************************************************
  //**********************************************************************
  // TRAINING FINISHED .................................................
  //
  // Let's store the network, report the log

  if(trace&1) TraceLog("Training finished");

  //write the network
  if (NULL != p_targetmmf) {
    if(trace&1) TraceLog(std::string("Writing network: ")+p_targetmmf);
    network.WriteNetwork(p_targetmmf);
  } else {
    Error("missing argument --TARGETMMF");
  }

  timer.End();
  std::cout << "===== TRbmCu FINISHED ( " << timer.Val() << "s ) "
            << "[FPS:" << mse.GetFrames() / timer.Val() 
            << ",RT:" << 1.0f / (mse.GetFrames() / timer.Val() / 100.0f)
            << "] =====" << std::endl;

  //report objective function (accuracy, frame counts...)
  std::cout << mse.Report();

  if(trace &4) {
    std::cout << "\n== PROFILE ==\nT-fe: " << time_frontend << std::endl;
  }
  
  return  0; ///finish OK

} catch (std::exception& rExc) {
  std::cerr << "Exception thrown" << std::endl;
  std::cerr << rExc.what() << std::endl;
  return  1;
}
