
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

#define SVN_DATE       "$Date: 2011-04-04 19:14:16 +0200 (Mon, 04 Apr 2011) $"
#define SVN_AUTHOR     "$Author: iveselyk $"
#define SVN_REVISION   "$Revision: 46 $"
#define SVN_ID         "$Id: TNet.cc 46 2011-04-04 17:14:16Z iveselyk $"

#define MODULE_VERSION "1.0.0 "__TIME__" "__DATE__" "SVN_ID  



/*** TNetLib includes */
#include "Error.h"
#include "Timer.h"
#include "Features.h"
#include "Common.h"
#include "MlfStream.h"
#include "UserInterface.h"
#include "Timer.h"

/*** TNet includes */
#include "Nnet.h"
#include "ObjFun.h"
#include "Platform.h"


/*** STL includes */
#include <iostream>
#include <sstream>
#include <numeric>


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
":TODO:\n\n"
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
"BUNCHSIZE CACHESIZE CONFUSIONMODE[no,max,soft,dmax,dsoft] CROSSVALIDATE FEATURETRANSFORM LEARNINGRATE LEARNRATEFACTORS MLFTRANSC MOMENTUM NATURALREADORDER OBJECTIVEFUNCTION[mse,xent] OUTPUTLABELMAP PRINTCONFIG PRINTVERSION RANDOMIZE SCRIPT SEED SOURCEMLF SOURCEMMF SOURCETRANSCDIR SOURCETRANSCEXT TARGETMMF TARGETMODELDIR TARGETMODELEXT TRACE WEIGHTCOST\n"
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


int main(int argc, char *argv[])
{
  const char* p_option_string =
    " -c n   CROSSVALIDATE=TRUE"
//  " -d r   SOURCEMODELDIR"
    " -m r   OUTPUTLABELMAP" 
    " -n r   LEARNINGRATE" 
    " -o r   TARGETMODELEXT" 
    " -p r   PARALLELMODE" 
//  " -r n   REGULARISATION=TRUE" //add later
//  " -u r   UPDATE" //add later, update only certain weights...
//  " -x r   SOURCEMODELEXT"
    " -B n   SAVEBINARY=TRUE" 
    " -D n   PRINTCONFIG=TRUE"
//  " -G r   SOURCETRANSCFMT" //add if more transcription formats
    " -H l   SOURCEMMF"
    " -I r   SOURCEMLF"
    " -L r   SOURCETRANSCDIR"
    " -M r   TARGETMODELDIR"
    " -O r   OBJECTIVEFUNCTION" 
    " -S l   SCRIPT"
    " -T r   TRACE"
    " -V n   PRINTVERSION=TRUE"
    " -X r   SOURCETRANSCEXT";


  try {
    UserInterface        ui;
    Platform             pl;
    Timer                timer;

   
    const char*                       p_script;
    const char*                       p_output_label_map;
    BaseFloat                         learning_rate;
    BaseFloat                         weightcost;
    ObjectiveFunction::ObjFunType     obj_fun_id;
    CrossEntropy::ConfusionMode       xent_conf_mode;

    const char*                       p_source_mmf_file;
    const char*                       p_input_transform;

    const char*                       p_targetmmf; //< SNet legacy --TARGETMMF
          char                        p_trg_mmf_file[4096];
    const char*                       p_trg_mmf_dir;
    const char*                       p_trg_mmf_ext;

    const char*                       p_source_mlf_file;
    const char*                       p_src_lbl_dir;
    const char*                       p_src_lbl_ext;

    int                               bunch_size;
    int                               cache_size;
    bool                              randomize;
    long int                          seed;


    int                               trace;
    bool                              crossval;
    int                               num_threads;


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
    
    p_targetmmf         = ui.GetStr(SNAME":TARGETMMF",     NULL);//< has higher priority than "dir/file.ext" composition (SNet legacy)
    p_trg_mmf_dir       = ui.GetStr(SNAME":TARGETMODELDIR",  "");//< dir for composition
    p_trg_mmf_ext       = ui.GetStr(SNAME":TARGETMODELEXT",  "");//< ext for composition

    p_script            = ui.GetStr(SNAME":SCRIPT",         NULL);
    p_output_label_map  = ui.GetStr(SNAME":OUTPUTLABELMAP", NULL);
    learning_rate       = ui.GetFlt(SNAME":LEARNINGRATE"  , 0.06f);
    weightcost          = ui.GetFlt(SNAME":WEIGHTCOST"    , 0.0);

    obj_fun_id          = static_cast<ObjectiveFunction::ObjFunType>(
                          ui.GetEnum(SNAME":OBJECTIVEFUNCTION", 
                                     ObjectiveFunction::CROSS_ENTROPY, //< default
                                     "ent", ObjectiveFunction::CROSS_ENTROPY,
                                     "mse", ObjectiveFunction::MEAN_SQUARE_ERROR
                          ));

    xent_conf_mode      = static_cast<CrossEntropy::ConfusionMode>(
                          ui.GetEnum(SNAME":CONFUSIONMODE",
                                     CrossEntropy::NO_CONF, //< default
                                     "no", CrossEntropy::NO_CONF,
                                     "max", CrossEntropy::MAX_CONF,
                                     "soft", CrossEntropy::SOFT_CONF,
                                     "dmax", CrossEntropy::DIAG_MAX_CONF,
                                     "dsoft", CrossEntropy::DIAG_SOFT_CONF
                          ));

    p_source_mlf_file   = ui.GetStr(SNAME":SOURCEMLF",       NULL);
    p_src_lbl_dir       = ui.GetStr(SNAME":SOURCETRANSCDIR", NULL);
    p_src_lbl_ext       = ui.GetStr(SNAME":SOURCETRANSCEXT", "lab");

    bunch_size          = ui.GetInt(SNAME":BUNCHSIZE", 256);
    cache_size          = ui.GetInt(SNAME":CACHESIZE", 12800);
    randomize           = ui.GetBool(SNAME":RANDOMIZE", true);

    //cannot get long int
    seed                = ui.GetInt(SNAME":SEED", 0);


    //Fill the global variables of the singleton 'Gl'
    trace               = ui.GetInt(SNAME":TRACE",               0);
    num_threads         = ui.GetInt(SNAME":THREADS",          1);
    crossval            = ui.GetBool(SNAME":CROSSVALIDATE",  false);


    // process the parameters
    if(ui.GetBool(SNAME":PRINTCONFIG", false)) {
      std::cout << std::endl;
      ui.PrintConfig(std::cout);
      std::cout << std::endl;
    }
    if(ui.GetBool(SNAME":PRINTVERSION", false)) {
      std::cout << std::endl;
      std::cout << "======= TNET v"MODULE_VERSION" =======" << std::endl;
      std::cout << std::endl;
    }
    ui.CheckCommandLineParamUse();
    

    // the rest of the parameters are the feature files
    for (; args_parsed < argc; args_parsed++) {
      pl.feature_.AddFile(argv[args_parsed]);
    }

    //**************************************************************************
    //**************************************************************************
    // OPTION PARSING DONE .....................................................


    //initialize the InputProxy
    if(NULL == p_script)
      Warning("WARNING: The script file is missing [-S]");
    if(NULL == p_source_mlf_file)
      Error("Source mlf file file is missing [-I]");
    if(NULL == p_output_label_map)
      Error("Output label map is missing [-m]");

    // initialize the feature repository
    if(trace&1) TraceLog("Initializing FeatureRepository");
    pl.feature_.Init(
      swap_features, start_frm_ext, end_frm_ext, target_kind,
      deriv_order, p_deriv_win_lenghts, 
      cmn_path, cmn_mask, cvn_path, cvn_mask, cvg_file
    );
    //open the scp file
    pl.feature_.AddFileList(p_script);

    // initialize the label repository
    if(trace&1) TraceLog("Initializing LabelRepository");
    pl.label_.Init(p_source_mlf_file,p_output_label_map, p_src_lbl_dir, p_src_lbl_ext);

    // read input transform    
    if(NULL != p_input_transform) {
      if(trace&1) TraceLog(std::string("Reading input transform:")+p_input_transform);
      pl.nnet_transf_.ReadNetwork(p_input_transform);
    }

    // read network
    if(NULL != p_source_mmf_file) { 
      if(trace&1) TraceLog(std::string("Reading network:")+p_source_mmf_file);
      pl.nnet_.ReadNetwork(p_source_mmf_file);
    } else {
      Error("Source MMF must be specified [-H]");
    }
    pl.nnet_.SetLearnRate(learning_rate);
    pl.nnet_.SetWeightcost(weightcost);

    //get objective function instance
    pl.obj_fun_ = ObjectiveFunction::Factory(obj_fun_id);
    //setup the cross entropy
    if(obj_fun_id == ObjectiveFunction::CROSS_ENTROPY) {
      CrossEntropy* xent = dynamic_cast<CrossEntropy*>(pl.obj_fun_);
      //confusion mode
      xent->SetConfusionMode(xent_conf_mode);
      //pass the outputlabelmap
      xent->SetOutputLabelMap(p_output_label_map);
    }

    //initialize the cache
    pl.bunchsize_ = bunch_size;
    pl.cachesize_ = cache_size;
    pl.randomize_ = randomize;
    //
    pl.start_frm_ext_ = start_frm_ext;
    pl.end_frm_ext_ = end_frm_ext;
    pl.trace_ = trace;
    pl.crossval_ = crossval;

    //TODO do someting with seed!!!
    pl.seed_ = seed;
    //data_proxy.InitCache(cache_size, bunch_size, network, randomize, seed);
    
    timer.Start();
    std::cout << "===== TNET " 
              << (crossval?"CROSSVALIDATION":"TRAINING") 
              << " STARTED =====" << std::endl;
    std::cout << "Objective function: " 
              << pl.obj_fun_->GetName() << std::endl;
    if(!crossval) {
      std::cout << "Learning rate: " << learning_rate << std::endl;
    }


    /*
     * PERFORM ONE ITERATION OF THE TRAINING
     */
    pl.RunTrain(num_threads);
    /*
     *
     */


    if(trace&1) TraceLog("Training finished");

    //write the network
    if(!crossval) {
      if(trace&1) TraceLog("Writing network");
      if (NULL != p_targetmmf) {
        pl.nnet_.WriteNetwork(p_targetmmf);
      } else {
        MakeHtkFileName(p_trg_mmf_file, p_source_mmf_file, p_trg_mmf_dir, p_trg_mmf_ext);
        pl.nnet_.WriteNetwork(p_trg_mmf_file);
      }
    }

    //show report
    timer.End();

    pl.cout_mutex_.Lock();

    std::cout << "===== TNET FINISHED ( " << timer.Val() << "s ) "
              << "[ FPS: " << pl.obj_fun_->GetFrames() / timer.Val() 
              << " RT: " << 1.0f / (pl.obj_fun_->GetFrames() / timer.Val() / 100.0f)
              << " ] =====" << std::endl;

    //report objective function
    std::cout << "-- " << (crossval?"CV ":"TR ") 
              << pl.obj_fun_->Report();

    pl.cout_mutex_.Unlock();

  }
  catch (std::exception& rExc) {
    std::cerr << "Exception thrown" << std::endl;
    std::cerr << rExc.what() << std::endl;
    return 1;
  }
  return 0;
}
