#include "Labels.h"
#include "Timer.h"


namespace TNet {


  ////////////////////////////////////////////////////////////////////////
  // Class LabelRepository::
  void
  LabelRepository::
  Init(const char* pLabelMlfFile, const char* pOutputLabelMapFile, const char* pLabelDir, const char* pLabelExt)
  {
    assert(NULL != pLabelMlfFile);
    assert(NULL != pOutputLabelMapFile);

    // initialize the label streams
    delete mpLabelStream; //if NULL, does nothing
    delete _mpLabelStream;
    _mpLabelStream = new std::ifstream(pLabelMlfFile);
    mpLabelStream  = new IMlfStream(*_mpLabelStream);

    // Label stream is initialized, just test it
    if(!mpLabelStream->good()) 
      Error(std::string("Cannot open Label MLF file: ")+pLabelMlfFile);

    // Index the labels (good for randomized file lists)
    Timer tim; tim.Start();
    mpLabelStream->Index();
    tim.End(); mIndexTime += tim.Val(); 

    // Read the state-label to state-id map
    ReadOutputLabelMap(pOutputLabelMapFile);

    // Store the label dir/ext
    mpLabelDir = pLabelDir;
    mpLabelExt = pLabelExt;
  }



  void 
  LabelRepository::
  GenDesiredMatrix(BfMatrix& rDesired, size_t nFrames, size_t sourceRate, const char* pFeatureLogical)
  {
    //timer
    Timer tim; tim.Start();
    
    //Get the MLF stream reference...
    IMlfStream& mLabelStream = *mpLabelStream;
    //Build the file name of the label
    MakeHtkFileName(mpLabelFile, pFeatureLogical, mpLabelDir, mpLabelExt);

    //Find block in MLF file
    mLabelStream.Open(mpLabelFile);
    if(!mLabelStream.good()) {
      Error(std::string("Cannot open label MLF record: ") + mpLabelFile);
    }


    //resize the matrix
    if(nFrames < 1) {
      KALDI_ERR << "Number of frames:" << nFrames << " is lower than 1!!!\n"
                << pFeatureLogical;
    }
    rDesired.Init(nFrames, mLabelMap.size(), true); //true: Zero()

    //aux variables
    std::string line, state;
    unsigned long long beg, end;
    size_t state_index;
    size_t trunc_frames = 0;
    TagToIdMap::iterator it;
    
    //parse the label file
    while(!mLabelStream.eof()) {
      std::getline(mLabelStream, line);
      if(line == "") continue; //skip newlines/comments from MLF
      if(line[0] == '#') continue;

      std::istringstream& iss = mGenDesiredMatrixStream;
      iss.clear();
      iss.str(line);

      //parse the line
      //begin
      iss >> std::ws >> beg;
      if(iss.fail()) { 
        KALDI_ERR << "Cannot parse column 1 (begin)\n"
                  << "line: " << line << "\n"
                  << "file: " << mpLabelFile << "\n";
      }
      //end
      iss >> std::ws >> end;
      if(iss.fail()) { 
        KALDI_ERR << "Cannot parse column 2 (end)\n"
                  << "line: " << line << "\n"
                  << "file: " << mpLabelFile << "\n";
      }
      //state tag
      iss >> std::ws >> state;
      if(iss.fail()) { 
        KALDI_ERR << "Cannot parse column 3 (state_tag)\n"
                  << "line: " << line << "\n"
                  << "file: " << mpLabelFile << "\n";
      }

      //divide beg/end by sourceRate and round up to get interval of frames
      beg = (beg+sourceRate/2)/sourceRate;
      end = (end+sourceRate/2)/sourceRate; 
      //beg = (int)round(beg / (double)sourceRate);
      //end = (int)round(end / (double)sourceRate); 
      
      //find the state id
      it = mLabelMap.find(state);
      if(mLabelMap.end() == it) {
        Error(std::string("Unknown state tag: '") + state + "' file:'" + mpLabelFile);
      }
      state_index = it->second;

      // Fill the desired matrix
      for(unsigned long long frame=beg; frame<end; frame++) { 
        //don't write after matrix... (possible longer transcript than feature file)
        if(frame >= (int)rDesired.Rows()) { trunc_frames++; continue; }

        //check the next frame is empty:
        if(0.0 != rDesired[frame].Sum()) {
          //ERROR!!!
          //find out what was previously filled!!!
          BaseFloat max = rDesired[frame].Max();
          int idx = -1;
          for(int i=0; i<(int)rDesired[frame].Dim(); i++) { 
            if(rDesired[frame][i] == max) idx = i; 
          }
          for(it=mLabelMap.begin(); it!=mLabelMap.end(); ++it) {
            if((int)it->second == idx) break;
          }
          std::string state_prev = "error";
          if(it != mLabelMap.end()) {
            state_prev = it->first;
          }
          //print the error message
          std::ostringstream os; 
          os << "Frame already assigned to other state, "
             << " file: " << mpLabelFile 
             << " frame: " << frame
             << " nframes: " << nFrames 
             << " sum: " << rDesired[frame].Sum()  
             << " previously assigned to: " << state_prev << "(" << idx << ")" 
             << " now should be assigned to: " << state << "(" << state_index << ")"
             << "\n";
          Error(os.str());
        }

        //fill the row
        rDesired[(size_t)frame][state_index] = 1.0f;
      }
    }

    mLabelStream.Close();

    //check the desired matrix (rows sum up to 1.0)
    for(size_t i=0; i<rDesired.Rows(); ++i) {
      float desired_row_sum = rDesired[i].Sum();
      if(!desired_row_sum == 1.0) {
        std::ostringstream os;
        os << "Desired vector sum isn't 1.0, "
           << " file: " << mpLabelFile 
           << " row: " << i 
           << " nframes: " << nFrames 
           << " content: " << rDesired[i] 
           << " sum: " << desired_row_sum  << "\n";
        Error(os.str());
      }
    }
    
    //warning when truncating many frames
    if(trunc_frames > 10) {
      std::ostringstream os;
      os << "Truncated frames: " << trunc_frames 
         << " Check sourcerate in features and validity of labels\n";
      Warning(os.str());
    }

    //timer
    tim.End(); mGenDesiredMatrixTime += tim.Val();
  }

  

  void
  LabelRepository::
  ReadOutputLabelMap(const char* file)
  {
    assert(mLabelMap.size() == 0);
    int i = 0;
    std::string state_tag;
    std::ifstream in(file);
    if(!in.good())
      Error(std::string("Cannot open OutputLabelMapFile: ")+file);

    in >> std::ws;
    while(!in.eof()) {
      in >> state_tag;
      in >> std::ws;
      assert(mLabelMap.find(state_tag) == mLabelMap.end());
      mLabelMap[state_tag] = i++;
    }

    in.close();
    assert(mLabelMap.size() > 0);
  }


}//namespace
