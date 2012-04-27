
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
#define SVN_ID         "$Id: TNormCu.cc 94 2011-12-08 10:59:03Z iveselyk $"

#define MODULE_VERSION "1.0.0 "__TIME__" "__DATE__" "SVN_ID  



/*** KaldiLib includes */
#include "Error.h"
#include "Timer.h"
#include "Features.h"
#include "Common.h"
#include "UserInterface.h"
#include "Timer.h"

/*** TNet includes */
#include "cuNetwork.h"
#include "Nnet.h"

/*** STL includes */
#include <iostream>
#include <sstream>
#include <numeric>




//////////////////////////////////////////////////////////////////////
// DEFINES
//

#define SNAME "TNORM"

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
" -A         Print command line arguments                    Off\n" 
" -C cf      Set config file to cf                           Default\n"
" -D         Display configuration variables                 Off\n"
" -H mmf     Load NN macro file                              \n"
" -S file    Set script file                                 None\n"
" -T N       Set trace flags to N                            0\n" 
" -V         Print version information                       Off\n"
"\n"
"NATURALREADORDER PRINTCONFIG PRINTVERSION SCRIPT SOURCEMMF TARGETMMF TRACE\n"
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
    " -D n   PRINTCONFIG=TRUE"
    " -H l   SOURCEMMF"
    " -S l   SCRIPT"
    " -T r   TRACE"
    " -V n   PRINTVERSION=TRUE"
    ;


  UserInterface        ui;
  FeatureRepository    features;
  CuNetwork            network;
  Network              network_cpu;
  Timer                timer;

 
  const char*                       p_script;
  const char*                       p_source_mmf_file;
  const char*                       p_targetmmf; 

  int traceFlag;


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
  p_targetmmf         = ui.GetStr(SNAME":TARGETMMF",     NULL);//< target for mean/variance

  p_script            = ui.GetStr(SNAME":SCRIPT",         NULL);

  traceFlag       = ui.GetInt(SNAME":TRACE",               0);
  if(traceFlag&1) { CuDevice::Instantiate().Verbose(true); }


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

  //read the neural network
  if(NULL != p_source_mmf_file) { 
    if(CuDevice::Instantiate().IsPresent()) {
      if(traceFlag&1) TraceLog(std::string("Reading GPU network: ")+p_source_mmf_file);
      network.ReadNetwork(p_source_mmf_file);
    } else {
      if(traceFlag&1) TraceLog(std::string("Reading CPU network: ")+p_source_mmf_file);
      network_cpu.ReadNetwork(p_source_mmf_file);
    }
  } else {
    Error("Source MMF must be specified [-H]");
  }




  // initialize the feature repository 
  features.Init(
    swap_features, start_frm_ext, end_frm_ext, target_kind,
    deriv_order, p_deriv_win_lenghts, 
    cmn_path, cmn_mask, cvn_path, cvn_mask, cvg_file
  );
  if(NULL != p_script) {
    features.AddFileList(p_script);
  } else {
    Warning("WARNING: The script file is missing [-S]");
  }


  
  
  //**********************************************************************
  //**********************************************************************
  // INITIALIZATION DONE .................................................
  //
  // Start training
  timer.Start();
  std::cout << "===== TNormCu STARTED =====" << std::endl;

  int dim = CuDevice::Instantiate().IsPresent() ?
                network.GetNOutputs() :
                network_cpu.GetNOutputs();

  Vector<double> first(dim); first.Set(0.0);
  Vector<double> second(dim); second.Set(0.0);

  unsigned long framesN = 0;
 
  //progress
  size_t cnt = 0;
  size_t step = features.QueueSize() / 100;
  if(step == 0) step = 1;
 
  //**********************************************************************
  //**********************************************************************
  // MAIN LOOP

  for(features.Rewind(); !features.EndOfList(); features.MoveNext()) {

    Matrix<BaseFloat> feats_host,net_out;
    Matrix<BaseFloat> feats_host_out;
    CuMatrix<BaseFloat> feats;
    CuMatrix<BaseFloat> feats_expanded;
  
    //get features 
    features.ReadFullMatrix(feats_host);

    if(CuDevice::Instantiate().IsPresent()) {
      //propagate 
      feats.CopyFrom(feats_host);
      network.Propagate(feats,feats_expanded);
      
      //trim the xxx_frm_ext
      int rows = feats_expanded.Rows()-start_frm_ext-end_frm_ext;
      CuMatrix<BaseFloat> feats_trim(rows,feats_expanded.Cols());
      feats_trim.CopyRows(rows,start_frm_ext,feats_expanded,0);
      feats_trim.CopyTo(feats_host_out);
    } else {
      //propagate
      network_cpu.Propagate(feats_host,net_out);
      //trim the xxx_frm_ext
      feats_host_out.Init(net_out.Rows()-start_frm_ext-end_frm_ext,net_out.Cols());
      memcpy(feats_host_out.pData(),net_out.pRowData(start_frm_ext),feats_host_out.MSize());
    }

    //accumulate first/second order statistics
    for(size_t m=0; m<feats_host_out.Rows(); m++) {
      for(size_t n=0; n<feats_host_out.Cols(); n++) {
        BaseFloat val = feats_host_out(m,n);
        first[n] += val; 
        second[n] += val*val;

        if(isnan(first[n])||isnan(second[n])||
           isinf(first[n])||isinf(second[n])) 
        {
          std::ostringstream oss;
          oss << "nan/inf in accumulators\n"
              << "first:" << first << "\n"
              << "second:" << second << "\n"
              << "frames:" << framesN << "\n"
              << "utterance:" << features.Current().Logical() << "\n"
              << "feats_host: " << feats_host << "\n"
              << "feats_host_out: " << feats_host_out << "\n";
          Error(oss.str());
        }
      }
    }

    

    framesN += feats_host.Rows();
    
    //progress 
    if((cnt++ % step) == 0) std::cout << 100 * cnt / features.QueueSize() << "%, " << std::flush;
  }

  //**********************************************************************
  //**********************************************************************
  // ACCUMULATING FINISHED .................................................
  //


  //get the mean/variance vectors
  Vector<double> mean(first);
  mean.Scale(1.0/framesN);
  Vector<double> variance(second);
  variance.Scale(1.0/framesN);
  for(size_t i=0; i<mean.Dim(); i++) {
    variance[i] -= mean[i]*mean[i];
  }

  //get the mean normalization biase vector, 
  //use negative mean vector
  Vector<double> bias(mean);
  bias.Scale(-1.0);

  //get the variance normalization window vector, 
  //inverse of square root of variance
  Vector<double> window(variance);
  for(size_t i=0; i<window.Dim(); i++) {
    window[i] = 1.0/sqrt(window[i]);
  }

  //store the normalization network
  std::ofstream os(p_targetmmf);
  if(!os.good()) Error(std::string("Cannot open file for writing: ")+p_targetmmf);

  dim = mean.Dim();
  os << "<bias> " << dim << " " << dim << "\n"
     << bias << "\n\n"
     << "<window> " << dim << " " << dim << "\n"
     << window << "\n\n";

  os.close();

  timer.End();
  std::cout << "\n\n===== TNormCu FINISHED ( " << timer.Val() << "s ) "
            << "[FPS:" << framesN / timer.Val() 
            << ",RT:" << 1.0f / (framesN / timer.Val() / 100.0f)
            << "] =====" << std::endl;

  std::cout << "frames: " << framesN 
            << ", max_bias: " << bias.Max()
            << ", max_window: " << window.Max()
            << ", min_window: " << window.Min()
            << "\n";
  
  return  0; ///finish OK

} catch (std::exception& rExc) {
  std::cerr << "Exception thrown" << std::endl;
  std::cerr << rExc.what() << std::endl;
  return  1;
}

