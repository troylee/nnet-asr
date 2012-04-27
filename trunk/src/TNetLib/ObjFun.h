#ifndef _TNET_OBJ_FUN_H
#define _TNET_OBJ_FUN_H

#include <cassert>
#include <limits>
#include <cmath>

#include "Matrix.h"
#include "Vector.h"

namespace TNet {

  /**
   * General interface for objective functions
   */
  class ObjectiveFunction
  {
    public:
    /// Enum with objective function types
    typedef enum { 
      OBJ_FUN_I = 0x0300, 
      MEAN_SQUARE_ERROR, 
      CROSS_ENTROPY, 
    } ObjFunType;
    
    public:
      /// Factory for creating objective function instances
      static ObjectiveFunction* Factory(ObjFunType type);
    
    //////////////////////////////////////////////////////////////
    // Interface specification
    protected:
      ObjectiveFunction() { }; /// constructor
    public:
      virtual ~ObjectiveFunction() { };  /// destructor

      virtual ObjFunType GetType() = 0;
      virtual const char* GetName() = 0;
      virtual ObjectiveFunction* Clone() = 0; 

      ///calculate error of network output
      virtual void Evaluate(const Matrix<BaseFloat>& net_out, const Matrix<BaseFloat>& target, Matrix<BaseFloat>* err) = 0;
 
      ///get the accumulated error
      virtual double GetError() = 0;
      ///the number of processed frames 
      virtual size_t GetFrames() = 0;
       
      ///report the error to string 
      virtual std::string Report() = 0;     

      ///sum the frame counts from more instances
      virtual void MergeStats(const ObjectiveFunction& inst) = 0;
  };



  /**
   * Mean square error function
   */
  class MeanSquareError : public ObjectiveFunction
  {
   public:
    MeanSquareError()
     : ObjectiveFunction(), frames_(0), error_(0)
    { }

    ~MeanSquareError()
    { }

    ObjFunType GetType()
    { return MEAN_SQUARE_ERROR; }

    const char* GetName()
    { return "<MeanSquareError>"; }

    ObjectiveFunction* Clone()
    { return new MeanSquareError(*this); }
    
    void Evaluate(const Matrix<BaseFloat>& net_out, const Matrix<BaseFloat>& target, Matrix<BaseFloat>* err);

    size_t GetFrames()
    { return frames_; }
    
    double GetError()
    { return error_; }

    std::string Report();    
     
    void MergeStats(const ObjectiveFunction& inst) { 
      const MeanSquareError& mse = dynamic_cast<const MeanSquareError&>(inst);
      frames_ += mse.frames_; error_ += mse.error_; 
    }
   
   private:
    size_t frames_;
    double error_;

  };


  /**
   * Cross entropy error function
   */
  class CrossEntropy : public ObjectiveFunction
  {
   public:
    enum ConfusionMode { NO_CONF=0, MAX_CONF, SOFT_CONF, DIAG_MAX_CONF, DIAG_SOFT_CONF };

   public:
    CrossEntropy()
     : ObjectiveFunction(), frames_(0), error_(0), corr_(0), confusion_mode_(NO_CONF), output_label_map_(NULL)
    { }

    ~CrossEntropy()
    { }

    ObjFunType GetType()
    { return CROSS_ENTROPY; }

    const char* GetName() 
    { return "<cross_entropy>"; }

    ObjectiveFunction* Clone()
    { return new CrossEntropy(*this); }

    void Evaluate(const Matrix<BaseFloat>& net_out, const Matrix<BaseFloat>& target, Matrix<BaseFloat>* err);

    size_t GetFrames()
    { return frames_; }
    
    double GetError()
    { return error_; }

    void SetConfusionMode(enum ConfusionMode m)
    { confusion_mode_ = m; }

    void SetOutputLabelMap(const char* map)
    { output_label_map_ = map; }

    std::string Report();    
     
    void MergeStats(const ObjectiveFunction& inst);   
   private:
    size_t frames_;
    double error_;
    size_t corr_;
 
    ConfusionMode confusion_mode_;
    Matrix<float> confusion_;
    Vector<int> confusion_count_;
    Vector<double> diag_confusion_;
    const char* output_label_map_;
  };
 

} //namespace TNet


#endif
