
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

#define SVN_DATE       "$Date: 2011-04-29 14:18:20 +0200 (Fri, 29 Apr 2011) $"
#define SVN_AUTHOR     "$Author: iveselyk $"
#define SVN_REVISION   "$Revision: 49 $"
#define SVN_ID         "$Id: TMpeCu.cc 49 2011-04-29 12:18:20Z iveselyk $"

#define MODULE_VERSION "1.0.0 "__TIME__" "__DATE__" "SVN_ID  



/*** STK includes */
#include "STKLib/trunk/config.h"
#ifdef HAVE_MEMALIGN
  #undef HAVE_MEMALIGN
#endif
#ifdef HAVE_POSIX_MEMALIGN
  #undef HAVE_POSIX_MEMALIGN
#endif


/*** include commons */
#include "STKLib/common.h"


#include "Common.h"

/*** STK includes */
#include "STKLib/fileio.h"
#include "STKLib/Models.h"
#include "STKLib/Decoder.h"
#include "STKLib/stkstream.h"
#include "STKLib/MlfStream.h"
#include "STKLib/labels.h"


/*** Kaldi includes */
#include "Error.h"
#include "Timer.h"
#include "Features.h"
#include "UserInterface.h"


/*** TNet includes */
#include "cuObjectiveFunction.h"
#include "cuNetwork.h"
#include "cuCache.h"
#include "cuda.h"


/*** STL includes */
#include <iostream>
#include <sstream>
#include <numeric>




//////////////////////////////////////////////////////////////////////
// DEFINES
//

#define SNAME "TMPECU"

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
" -t f [i l] Set pruning to f [inc limit]                    Off\n"
" -A         Print command line arguments                    Off\n" 
" -C cf      Set config file to cf                           Default\n"
" -D         Display configuration variables                 Off\n"
" -G fmt     Set source trascription format to fmt           As config\n"
" -H mmf     Load NN macro file                              \n"
" -I mlf     Load master label file mlf (with den_num latts) \n"
" -L dir     Set input label (or net) dir                    Current\n"
//" -O fn      Objective function [mpe,mmi]                  mpe\n"
" -S file    Set script file                                 None\n"
" -T N       Set trace flags to N                            0\n" 
" -V         Print version information                       Off\n"
" -X ext     Set input label file ext                        lab\n"
"\n"
"ALLOWXWRDEXP ENDTIMESHIFT EXACTTIMEMERGE FEATURETRANSFORM GRADDIVFRM HMM HNETFILTER LEARNINGRATE LEARNRATEFACTORS LMSCALE MAXACTIVEMODELS MINACTIVEMODELS MINIMIZENET MLGAMMA MODELPENALTY NATURALREADORDER NFRAMEOUTPNORM OCCUPPSCALE OUTPSCALE POSTERIORSCALE PRINTCONFIG PRINTVERSION PRONUNSCALE PRUNING PRUNINGINC PRUNINGMAX REMEXPWRDNODES RESPECTPRONVARS SCRIPT SHOWGAMMA SOURCEDICT SOURCEMLF SOURCEMMF SOURCETRANSCDIR SOURCETRANSCEXT STARTTIMESHIFT TARGETMMF TIMEPRUNING TRACE TRANSPSCALE WEIGHTCOST WEIGHTPUSHING WORDPENALTY\n"
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
    " -t ror PRUNING PRUNINGINC PRUNINGMAX"
    " -D n   PRINTCONFIG=TRUE"
    " -G r   SOURCETRANSCFMT"
    " -H l   SOURCEMMF"
    " -I r   SOURCEMLF"
    " -L r   SOURCETRANSCDIR"
    " -S l   SCRIPT"
    " -T r   TRACE"
    " -V n   PRINTVERSION=TRUE"
    " -X r   SOURCETRANSCEXT";

  //STK global objects
  STK::ModelSet                        hset;
  STK::Decoder<STK::DecoderNetwork>    decoder;
  std::ostringstream                   os_warn;

  //TNet global objects
  UserInterface        ui;
  FeatureRepository    feature_repo;
  CuNetwork            network;
  CuNetwork            transform_network;
  Timer                timer;
  Timer                timer_frontend;
  double               time_frontend = 0.0;
  Timer                timer_decoder;
  double               time_decoder = 0.0;

  // vars for STK
  const char*                       p_hmm_file;
  const char*                       p_src_mlf;

  MyHSearchData                nonCDphHash;
  MyHSearchData                phoneHash;
  MyHSearchData                dictHash;

  double                            outprb_scale;
  char                              label_file[1024];
  FILE*                             ilfp = NULL;

  const char*                       src_lbl_dir;
  const char*                       src_lbl_ext;

  const char*                       dictionary;

  double                            word_penalty;
  double                            model_penalty;
  double                            grammar_scale;
  double                            posterior_scale;

  bool                              time_pruning;
  double                            pronun_scale;
  double                            transp_scale;
  double                            occprb_scale;
  double                            state_pruning;
  int                               max_active;
  int                               min_active;

  STK::ExpansionOptions             expOptions = {0};
  STKNetworkOutputFormat            in_net_fmt = {0};

  double                            stprn_step;
  double                            stprn_limit;

  STK::BasicVector<FLOAT>*          p_weight_vector = NULL;

  const char*                       net_filter;

  double                            avg_accuracy = 0.0;

  // vars for TNet 
  const char*                       p_script;

  BaseFloat                         learning_rate;
  const char*                       learning_rate_factors;
  BaseFloat                         weightcost;
  bool                              grad_div_frm;

  const char*                       p_source_mmf_file;
  const char*                       p_input_transform;

  const char*                       p_targetmmf;

  bool                              show_gamma;
  bool                              ml_gamma;

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


  // extract STK parameters
  p_hmm_file          = ui.GetStr(SNAME":HMM",     NULL);
  p_src_mlf           = ui.GetStr(SNAME":SOURCEMLF",     NULL);
  
  outprb_scale        = ui.GetFlt(SNAME":OUTPSCALE", 1.0);

  src_lbl_dir         = ui.GetStr(SNAME":SOURCETRANSCDIR", NULL);
  src_lbl_ext         = ui.GetStr(SNAME":SOURCETRANSCEXT", NULL);

  dictionary   =   ui.GetStr(SNAME":SOURCEDICT",      NULL);
  
  word_penalty =   ui.GetFlt(SNAME":WORDPENALTY",     0.0);
  model_penalty=   ui.GetFlt(SNAME":MODELPENALTY",    0.0);
  grammar_scale=   ui.GetFlt(SNAME":LMSCALE",         1.0);
  posterior_scale= ui.GetFlt(SNAME":POSTERIORSCALE", 1.0);


  time_pruning = ui.GetBool(SNAME":TIMEPRUNING",     false);
  in_net_fmt.mNoTimes = !time_pruning;

  pronun_scale = ui.GetFlt(SNAME":PRONUNSCALE",     1.0);
  transp_scale = ui.GetFlt(SNAME":TRANSPSCALE",     1.0);
  occprb_scale = ui.GetFlt(SNAME":OCCUPPSCALE",     1.0);
  state_pruning= ui.GetFlt(SNAME":PRUNING",         0.0);
  max_active   = ui.GetInt(SNAME":MAXACTIVEMODELS", 0);
  min_active   = ui.GetInt(SNAME":MINACTIVEMODELS", 0);

  expOptions.mCDPhoneExpansion =
                   ui.GetBool(SNAME":ALLOWXWRDEXP",    false);
  expOptions.mRespectPronunVar
                 = ui.GetBool(SNAME":RESPECTPRONVARS", false);
  expOptions.mStrictTiming
                 = ui.GetBool(SNAME":EXACTTIMEMERGE",  false);
  expOptions.mNoWeightPushing
                 =!ui.GetBool(SNAME":WEIGHTPUSHING",   true);
  expOptions.mNoOptimization
                 =!ui.GetBool(SNAME":MINIMIZENET",     false);
  expOptions.mRemoveWordsNodes
                 = ui.GetBool(SNAME":REMEXPWRDNODES",  false);

  stprn_step   = ui.GetFlt(SNAME":PRUNINGINC",      0.0);
  stprn_limit  = ui.GetFlt(SNAME":PRUNINGMAX",      0.0);

  net_filter   = ui.GetStr(SNAME":HNETFILTER",      NULL);
  if(NULL != net_filter) {
    transc_filter = net_filter;
  }

  in_net_fmt.mStartTimeShift =
                   ui.GetFlt(SNAME":STARTTIMESHIFT",  0.0);
  in_net_fmt.mEndTimeShift =
                   ui.GetFlt(SNAME":ENDTIMESHIFT",    0.0);




  // extract other parameters
  p_source_mmf_file   = ui.GetStr(SNAME":SOURCEMMF",       NULL);
  p_input_transform   = ui.GetStr(SNAME":FEATURETRANSFORM",NULL);
  
  p_targetmmf         = ui.GetStr(SNAME":TARGETMMF",       NULL);

  p_script            = ui.GetStr(SNAME":SCRIPT",          NULL);

  learning_rate       = ui.GetFlt(SNAME":LEARNINGRATE"  , 0.06f);
  learning_rate_factors = ui.GetStr(SNAME":LEARNRATEFACTORS", NULL);
  weightcost         = ui.GetFlt(SNAME":WEIGHTCOST"  , 0.0f);
  grad_div_frm        = ui.GetBool(SNAME":GRADDIVFRM",     true);

  show_gamma          = ui.GetBool(SNAME":SHOWGAMMA",     false);
  ml_gamma            = ui.GetBool(SNAME":MLGAMMA",       false);

  trace               = ui.GetInt(SNAME":TRACE",              0);
  if(trace&1) { CuDevice::Instantiate().Verbose(true); }




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
    feature_repo.AddFile(argv[args_parsed]);
  }

  //**************************************************************************
  //**************************************************************************
  // OPTION PARSING DONE .....................................................


  ////////////////////////////////////////////////////////////////////////////
  // initialize STK
  
  // initialize basic ModelSet
  hset.Init(STK::MODEL_SET_WITH_ACCUM);
  hset.mUpdateMask = 0;

  if (NULL != p_hmm_file) {
    TraceLog(std::string("Reading HMM model:")+p_hmm_file);
    hset.ParseMmf(p_hmm_file, NULL, false);
  } else {
    Error("Missing HMM model, use: --HMM=FILE");
  }
  
  hset.ExpandPredefXforms();
  hset.AttachPriors(&hset);

  nonCDphHash = hset.MakeCIPhoneHash();

  hset.mCmllrStats = false;
  hset.AllocateAccumulatorsForXformStats();


  hset.mUpdateType          = STK::UT_EBW;
  hset.mMinVariance         = 0.0;    ///< global minimum variance floor
  hset.MMI_E                = 2.0;
  hset.MMI_h                = 2.0;
  hset.MMI_tauI             = 200.0;
  hset.JSmoothing           = false;
  hset.mISmoothingMaxOccup  = -1.0;
  hset.mMinOccupation       = 0.0;
  hset.mMapTau              = 0;
  hset.mGaussLvl2ModelReest = false;
  hset.mMinOccurances       = 3;
  hset.mMinMixWeight        = 1.0 * MIN_WEGIHT;
  hset.mUpdateMask          = 0;
  hset.mSaveGlobOpts        = true;
  hset.mModelUpdateDoesNotNormalize      = false;
  hset.ResetAccums();  

  //open mlf with lattices
  ilfp = OpenInputMLF(p_src_mlf);

  //reserve space for hashes
  if (!STK::my_hcreate_r(100,  &dictHash) 
    || !STK::my_hcreate_r(100,  &phoneHash))
  {
    Error("Insufficient memory");
  }

  //read dictionary
  if (dictionary != NULL) {
    ReadDictionary(dictionary, &dictHash, &phoneHash);
  }
  if (dictHash.mNEntries == 0) 
    expOptions.mNoWordExpansion = 1;

  ////////////////////////////////////////////////////////////////////////////
  // initialize TNet

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

  //set the learnrate
  network.SetLearnRate(learning_rate, learning_rate_factors);
  
  //set the L2 regularization constant
  network.SetWeightcost(weightcost);

  //set division of gradient by number of frames
  network.SetGradDivFrm(grad_div_frm);


  
  
  //**********************************************************************
  //**********************************************************************
  // INITIALIZATION DONE .................................................
  //
  // Start training
  timer.Start();
  std::cout << "===== TMpeCu TRAINING STARTED =====" << std::endl;

  feature_repo.Rewind();
  
  //**********************************************************************
  //**********************************************************************
  // MAIN LOOP
  //
  int frames = 0;
  int done = 0;
  CuMatrix<BaseFloat> feats, posteriors, globerr;
  for(feature_repo.Rewind(); !feature_repo.EndOfList(); feature_repo.MoveNext()) {
    
    timer_frontend.Start();
      
    Matrix<BaseFloat> feats_host, posteriors_host, globerr_host;
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

    //forward pass
    network.Propagate(feats,posteriors);
    posteriors.CopyTo(posteriors_host);
    posteriors_host.ApplyLog();

    /***************************************************
     *************************************************** 
     * DECODER PART get the error derivatives
     *
     */
    {
      timer_decoder.Start();

      STK::Matrix<BaseFloat> posteriors_stk, gammas_stk;

      //copy the posteriors to STK matrix
      posteriors_stk.Init(posteriors_host.Rows(), posteriors_host.Cols());
      for(size_t r=0; r<posteriors_host.Rows(); r++) {
        memcpy(posteriors_stk[r], posteriors_host.pRowData(r),
           posteriors_host.Cols()*sizeof(BaseFloat));
      }

      //check dims
      if (hset.mInputVectorSize != posteriors_stk.Cols()) {
        std::ostringstream os;
        os <<"Vector size ["<<posteriors_stk.Cols()<<"]"
           <<" in '"<<feature_repo.Current().Logical()<<"'"
           <<" is incompatible with source HMM set ["<<hset.mInputVectorSize<<"]";
        Error(os.str());
      }

      //load lattice
      strcpy(label_file, feature_repo.Current().Logical().c_str());
      
      ilfp = OpenInputLabelFile(
              label_file, 
              src_lbl_dir,
              src_lbl_ext ? src_lbl_ext : "net",
              ilfp, 
              p_src_mlf);

      ReadSTKNetwork(
              ilfp, 
              &dictHash, 
              &phoneHash, 
              STK::WORD_NOT_IN_DIC_WARN, 
              in_net_fmt,
              feature_repo.CurrentHeader().mSamplePeriod, 
              label_file, 
              p_src_mlf, false, decoder.rNetwork());
      
      decoder.rNetwork().ExpansionsAndOptimizations(
            expOptions,
            in_net_fmt,
            &dictHash,
            &nonCDphHash,
            &phoneHash,
            word_penalty,
            model_penalty,
            grammar_scale,
            posterior_scale);

   //   CloseInputLabelFile(ilfp, p_src_mlf);
      
      //initialize the decoder
      decoder.Init(&hset, &hset);
       
      decoder.mTimePruning     = time_pruning;
      decoder.mWPenalty        = word_penalty;
      decoder.mMPenalty        = model_penalty;
      decoder.mLmScale         = grammar_scale;
      decoder.mPronScale       = pronun_scale;
      decoder.mTranScale       = transp_scale;
      decoder.mOutpScale       = outprb_scale;
      decoder.mOcpScale        = occprb_scale;
      decoder.mPruningThresh   = state_pruning > 0.0 ? state_pruning : -LOG_0;
      decoder.mMaxActiveModels = max_active;
      decoder.mMinActiveModels = min_active;
      decoder.mAccumType       = STK::AT_MPE;

      if(ml_gamma) {
        decoder.mAccumType = STK::AT_ML;
      }

      //decode
      double prn_step   = stprn_step;
      double prn_limit  = stprn_limit;

      int n_frames = (int)posteriors_stk.Rows();
      if (ui.GetBool(SNAME":NFRAMEOUTPNORM", false)) 
      {
        decoder.mOutpScale  = outprb_scale / n_frames;
        decoder.mPruningThresh /= n_frames;
        prn_step       /= n_frames;
        prn_limit      /= n_frames;
      }
      
      if(n_frames < 1) {
        Error(std::string("No posterior frames, ")+feature_repo.Current().Logical());
      }

      FLOAT P;
      FLOAT avgAcc;
      for (;;) 
      {
        //***** RUN FWBW with MPE, return gamma values *********/
        P = decoder.GetMpeGamma(posteriors_stk,gammas_stk, avgAcc,
              n_frames, feature_repo.Current().Weight(), p_weight_vector);

        if(P > LOG_MIN)
          break;

        if (decoder.mPruningThresh <= LOG_MIN ||
          prn_step <= 0.0 ||
          (decoder.mPruningThresh += prn_step) > prn_limit) 
        {
          Error(std::string("Overpruning or bad data, skipping file " +
             feature_repo.Current().Logical()));
          break;
        }
        
        os_warn.clear();
        os_warn << "Overpruning or bad data in file " << feature_repo.Current().Logical()
                << ", trying pruning threshold: " << decoder.mPruningThresh;
        Warning(os_warn.str());
      }
      avg_accuracy += avgAcc;
      
      //cleanup
      posteriors_stk.Destroy();
      decoder.Clear();

      //copy gammas to TNet matrix
      globerr_host.Init(gammas_stk.Rows(),gammas_stk.Cols());
      for(size_t r=0; r<posteriors_host.Rows(); r++) {
        memcpy(globerr_host.pRowData(r), gammas_stk[r],
           gammas_stk.Cols()*sizeof(BaseFloat));
      }
      
      //print gamma matrix for debug
      if(show_gamma) {
        std::cout << globerr_host;
      }

      //scale gammas by negative acoustic scale kapa
      // dE/d_activation = kapa(gama_den - gama_num) = -kapa(gama_mpe)
      globerr_host.Scale(-outprb_scale);
      //globerr_host.Scale(outprb_scale);

      timer_decoder.End(); time_decoder += timer_decoder.Val(); 
    }
    /**DECODER PART END********************************
     **************************************************/


    globerr.CopyFrom(globerr_host);

    //check the dimensionalities
    if(globerr.Rows() != posteriors.Rows()) {
      std::ostringstream os;
      os << "Non-matching number of rows," 
         << " netout:" << posteriors.Rows() 
         << " errfile:" << globerr.Rows();
      Error(os.str());
    }
    if(globerr.Cols() != posteriors.Cols()) {
      std::ostringstream os;
      os << "Non-matching number of network outputs," 
         << " netout:" << posteriors.Cols() 
         << " errfile:" << globerr.Cols();
      Error(os.str());
    }

    if(learning_rate != 0.0) {
      //backward pass
      network.Backpropagate(globerr);
    }

    frames += feats.Rows();
    if(trace&1 && (++done%100)==1) {
      std::cout << "(" << done << "/" << feature_repo.QueueSize() << ") ";
    }
   
    /* 
    unsigned int free, total;
    cuMemGetInfo(&free, &total);
    std::cout << "freemem:" << free / (1024*1024) << "MB "; 
    */
  }

  CloseInputMLF(ilfp);

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
    Error("forgot to specify --TARGETMMF argument");
  }
  
  timer.End();
  std::cout << "===== TMpeCu FINISHED ( " << timer.Val() << "s ) "
            << "[FPS:" << float(frames) / timer.Val() 
            << ",RT:" << 1.0f / (float(frames) / timer.Val() / 100.0f)
            << "] =====" << std::endl;

  std::cout << "-- MPE average approximate accuracy: "
            << avg_accuracy/(float)feature_repo.QueueSize()
            << " utterances: " << feature_repo.QueueSize()
            << std::endl;
  std::cout << "T-fe: " << time_frontend << std::endl;
  std::cout << "T-decode: " << time_decoder << std::endl;

  
  return  0; ///finish OK

} catch (std::exception& rExc) {
  std::cerr << "Exception thrown" << std::endl;
  std::cerr << rExc.what() << std::endl;
  return 1;
}
