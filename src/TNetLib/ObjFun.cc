
#include "ObjFun.h"
#include "Error.h"

#include <limits>

namespace TNet {


ObjectiveFunction* ObjectiveFunction::Factory(ObjFunType type) {
  ObjectiveFunction* ret = NULL;
  switch(type) {
    case MEAN_SQUARE_ERROR: ret = new MeanSquareError;    break;
    case CROSS_ENTROPY:     ret = new CrossEntropy;       break;
    default: Error("Unknown ObjectiveFunction type");
  }
  return ret;
}


/*
 * MeanSquareError
 */
void MeanSquareError::Evaluate(const Matrix<BaseFloat>& net_out, const Matrix<BaseFloat>& target, Matrix<BaseFloat>* err) {
  
  //check dimensions
  assert(net_out.Rows() == target.Rows());
  assert(net_out.Cols() == target.Cols());
  if(err->Rows() != net_out.Rows() || err->Cols() != net_out.Cols()) {
    err->Init(net_out.Rows(),net_out.Cols());
  }

  //compute global gradient
  err->Copy(net_out);
  err->AddScaled(-1,target);

  //compute loss function
  double sum = 0;
  for(size_t r=0; r<err->Rows(); r++) {
    for(size_t c=0; c<err->Cols(); c++) {
      BaseFloat val = (*err)(r,c);
      sum += val*val;
    }
  }
  error_ += sum/2.0;
  frames_ += net_out.Rows();
}


std::string MeanSquareError::Report() {
  std::stringstream ss;
  ss << "Mse:" << error_ << " frames:" << frames_
     << " err/frm:" << error_/frames_
     << "\n";
  return ss.str();
}


/*
 * CrossEntropy
 */

///Find maximum in float array
inline int FindMaxId(const BaseFloat* ptr, size_t N) {
  BaseFloat mval = -1e20f;
  int mid = -1;
  for(size_t i=0; i<N; i++) {
    if(ptr[i] > mval) {
      mid = i; mval = ptr[i];
    }
  }
  return mid;
}


void
CrossEntropy::Evaluate(const Matrix<BaseFloat>& net_out, const Matrix<BaseFloat>& target, Matrix<BaseFloat>* err)
{
  //check dimensions
  assert(net_out.Rows() == target.Rows());
  assert(net_out.Cols() == target.Cols());
  if(err->Rows() != net_out.Rows() || err->Cols() != net_out.Cols()) {
    err->Init(net_out.Rows(),net_out.Cols());
  }

  //allocate confunsion buffers
  if(confusion_mode_ != NO_CONF) {
    if(confusion_.Rows() != target.Cols() || confusion_.Cols() != target.Cols()) {
      confusion_.Init(target.Cols(),target.Cols());
      confusion_count_.Init(target.Cols());
      diag_confusion_.Init(target.Cols());
    }
  }

  //compute global gradient (assuming on softmax input)
  err->Copy(net_out);
  err->AddScaled(-1,target);

  //collect max values
  std::vector<size_t> max_target_id(target.Rows());
  std::vector<size_t> max_netout_id(target.Rows());
  //check correct classification
  int corr = 0;
  for(size_t r=0; r<net_out.Rows(); r++) {
    int id_netout = FindMaxId(net_out[r].pData(),net_out.Cols());
    int id_target = FindMaxId(target[r].pData(),target.Cols());
    if(id_netout == id_target) corr++;
    max_target_id[r] = id_target;//store the max value
    max_netout_id[r] = id_netout;
  }

  //compute loss function
  double sumerr = 0;
  for(size_t r=0; r<net_out.Rows(); r++) {
    if(target(r,max_target_id[r]) == 1.0) {
      //pick the max value..., rest is zero
      BaseFloat val = log(net_out(r,max_target_id[r]));
      if(val < -1e10f) val = -1e10f;
      sumerr += val;
    } else {
      //process whole posterior vect.
      for(size_t c=0; c<net_out.Cols(); c++) {
        if(target(r,c) != 0.0) {
          BaseFloat val = target(r,c)*log(net_out(r,c));
          if(val < -1e10f) val = -1e10f;
          sumerr += val;
        }
      }
    }
  }

  //accumulate confusuion network
  if(confusion_mode_ != NO_CONF) {
    for(size_t r=0; r<net_out.Rows(); r++) {
      int id_target = max_target_id[r];
      int id_netout = max_netout_id[r];
      switch(confusion_mode_) {
        case MAX_CONF:
          confusion_(id_target,id_netout) += 1;
          break;
        case SOFT_CONF:
          confusion_[id_target].Add(net_out[r]);
          break;
        case DIAG_MAX_CONF:
          diag_confusion_[id_target] += ((id_target==id_netout)?1:0);
          break;
        case DIAG_SOFT_CONF:
          diag_confusion_[id_target] += net_out[r][id_target];
          break;
        default:
          KALDI_ERR << "unknown confusion type" << confusion_mode_;
      }
      confusion_count_[id_target] += 1;
    }
  }

  error_ -= sumerr;
  frames_ += net_out.Rows();
  corr_ += corr;
}


std::string CrossEntropy::Report() {
  std::stringstream ss;
  ss << "Xent:" << error_ << " frames:" << frames_
     << " err/frm:" << error_/frames_
     << " correct[" << 100.0*corr_/frames_ << "%]"
     << "\n";

  if(confusion_mode_ != NO_CONF) {
    //read class tags
    std::vector<std::string> tag;
    { 
      std::ifstream ifs(output_label_map_);
      assert(ifs.good());
      std::string str;
      while(!ifs.eof()) {
        ifs >> str;
        tag.push_back(str);
      }
    }
    assert(confusion_count_.Dim() <= tag.size());

    //print confusion matrix
    if(confusion_mode_ == MAX_CONF || confusion_mode_ == SOFT_CONF) {
      ss << "Row:label Col:hyp\n" << confusion_ << "\n";
    }
    
    //***print per-target accuracies
    for(int i=0; i<confusion_count_.Dim(); i++) {
      //get the numerator
      BaseFloat numerator = 0.0;
      switch (confusion_mode_) {
        case MAX_CONF: case SOFT_CONF:
          numerator = confusion_[i][i];
          break;
        case DIAG_MAX_CONF: case DIAG_SOFT_CONF:
          numerator = diag_confusion_[i];
          break;
        default:
          KALDI_ERR << "Usupported confusion mode:" << confusion_mode_;
      }
      //add line to report
      ss << std::setw(30) << tag[i] << " " 
         << std::setw(10) << 100.0*numerator/confusion_count_[i] << "%" 
         << " [" << numerator << "/" << confusion_count_[i] << "]\n";
    } //***print per-target accuracies
  }// != NO_CONF

  return ss.str();
}


void CrossEntropy::MergeStats(const ObjectiveFunction& inst) { 
  const CrossEntropy& xent = dynamic_cast<const CrossEntropy&>(inst);
  frames_ += xent.frames_; error_ += xent.error_; corr_ += xent.corr_;
  //sum the confustion statistics
  if(confusion_mode_ != NO_CONF) {
    if(confusion_.Rows() != xent.confusion_.Rows()) {
      confusion_.Init(xent.confusion_.Rows(),xent.confusion_.Cols());
      confusion_count_.Init(xent.confusion_count_.Dim());
      diag_confusion_.Init(xent.diag_confusion_.Dim());
    }
    confusion_.Add(xent.confusion_);
    confusion_count_.Add(xent.confusion_count_);
    diag_confusion_.Add(xent.diag_confusion_);
  }
}
 

} // namespace TNet
