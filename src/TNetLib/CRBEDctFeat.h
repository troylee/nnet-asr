#ifndef _CUCRBEDCTFEATURES_H_
#define _CUCRBEDCTFEATURES_H_


#include "Component.h"
#include "Matrix.h"
#include "Vector.h"
#include "cblas.h"


namespace TNet {

  /**
   * Expands the time context of the input features
   * in N, out k*N, FrameOffset o_1,o_2,...,o_k
   * FrameOffset example 11frames: -5 -4 -3 -2 -1 0 1 2 3 4 5
   */
  class Expand : public Component
  {
   public:
    Expand(size_t nInputs, size_t nOutputs, Component* pPred)
      : Component(nInputs,nOutputs,pPred)
    { }

    ~Expand()
    { }

    ComponentType GetType() const
    { return EXPAND; }

    const char* GetName() const
    { return "<expand>"; }
   
    Component* Clone() const 
    { 
      Expand* p = new Expand(GetNInputs(),GetNOutputs(),NULL);
      p->mFrameOffset.Init(mFrameOffset.Dim()); 
      p->mFrameOffset.Copy(mFrameOffset); 
      return p; 
    }

    void ReadFromStream(std::istream& rIn)
    { rIn >> mFrameOffset; }

    void WriteToStream(std::ostream& rOut)  
    { rOut << mFrameOffset; }
     
   protected:
    void PropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y)
    {
      assert(X.Cols()*mFrameOffset.Dim() == Y.Cols());
      assert(X.Rows() == Y.Rows());

      for(size_t r=0;r<X.Rows();r++) {
        for(size_t off=0;off<mFrameOffset.Dim();off++) {
          int r_off = r + mFrameOffset[off];
          if(r_off < 0) r_off = 0;
          if(r_off >= X.Rows()) r_off = X.Rows()-1;
          memcpy(Y.pRowData(r)+off*X.Cols(),X.pRowData(r_off),sizeof(BaseFloat)*X.Cols());
        }
      }
    }

    void BackpropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y)
    { Error("__func__ Nonsense"); }

   protected:
    Vector<int> mFrameOffset;
  };



  /**
   * Rearrange the matrix columns according to the indices in mCopyFromIndices
   */
  class Copy : public Component
  {
   public:
    Copy(size_t nInputs, size_t nOutputs, Component* pPred)
      : Component(nInputs,nOutputs,pPred)
    { }

    ~Copy()
    { }

    ComponentType GetType() const
    { return COPY; }

    const char* GetName() const
    { return "<copy>"; }
    
    Component* Clone() const 
    { 
      Copy* p = new Copy(GetNInputs(),GetNOutputs(),NULL);
      p->mCopyFromIndices.Init(mCopyFromIndices.Dim()); 
      p->mCopyFromIndices.Copy(mCopyFromIndices); 
      return p; 
    }

    void ReadFromStream(std::istream& rIn)
    { 
      Vector<int> vec; rIn >> vec; vec.Add(-1); 
      mCopyFromIndices.Init(vec.Dim()).Copy(vec);
    }

    void WriteToStream(std::ostream& rOut)  
    { 
      Vector<int> vec(mCopyFromIndices); 
      vec.Add(1); rOut << vec; 
    }
     
   protected:
    void PropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y)
    {
      assert(mCopyFromIndices.Dim() == Y.Cols());
      for(int i=0; i<mCopyFromIndices.Dim();i++) {
        assert(mCopyFromIndices[i] >= 0 && mCopyFromIndices[i] < X.Cols());
      }
        
      for(size_t r=0; r<X.Rows(); r++) {
        for(size_t c=0; c<Y.Cols(); c++) {
          Y(r,c) = X(r,mCopyFromIndices[c]);
        }
      }
    }

    void BackpropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y)
    { Error("__func__ Nonsense"); }

   protected:
    Vector<int> mCopyFromIndices;
  };
  
  class Transpose : public Component
  {
   public:
    Transpose(size_t nInputs, size_t nOutputs, Component* pPred)
      : Component(nInputs,nOutputs,pPred), mContext(0)
    { }

    ~Transpose()
    { }

    ComponentType GetType() const
    { return TRANSPOSE; }

    const char* GetName() const
    { return "<transpose>"; }
 
    Component* Clone() const  
    { 
      Transpose* p = new Transpose(GetNInputs(),GetNOutputs(),NULL); 
      p->mCopyFromIndices.Init(mCopyFromIndices.Dim());
      p->mCopyFromIndices.Copy(mCopyFromIndices); 
      p->mContext = mContext;
      return p; 
    }
  
    void ReadFromStream(std::istream& rIn)
    { 
      rIn >> std::ws >> mContext;

      if(GetNInputs() != GetNOutputs()) { 
        Error("Input dim must be same as output dim"); 
      }
      
      Vector<int> vec(GetNInputs());
      int channels = GetNInputs() / mContext;
      for(int i=0, ch=0; ch<channels; ch++) {
        for(int idx=ch; idx < (int)GetNInputs(); idx+=channels, i++) {
          assert(i < (int)GetNInputs());
          vec[i] = idx;
        }
      }

      mCopyFromIndices.Init(vec.Dim()).Copy(vec); 
    }

    void WriteToStream(std::ostream& rOut)  
    { rOut << " " << mContext << "\n"; }
     
   protected:
    void PropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y)
    { 
      assert(mCopyFromIndices.Dim() == Y.Cols());
      for(int i=0; i<mCopyFromIndices.Dim();i++) {
        assert(mCopyFromIndices[i] >= 0 && mCopyFromIndices[i] < X.Cols());
      }
        
      for(size_t r=0; r<X.Rows(); r++) {
        for(size_t c=0; c<Y.Cols(); c++) {
          Y(r,c) = X(r,mCopyFromIndices[c]);
        }
      }
    }

    void BackpropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y)
    { Error("__func__ Nonsense"); }

   protected:
    int mContext;
    Vector<int> mCopyFromIndices;
  };


  /**
   * BlockLinearity is used for the blockwise multiplication by 
   * DCT transform loaded from disk
   */
  class BlockLinearity : public Component
  {
    public:
      BlockLinearity(size_t nInputs, size_t nOutputs, Component* pPred)
        : Component(nInputs,nOutputs,pPred)
      { }

      ~BlockLinearity()
      { }


      ComponentType GetType() const
      { return Component::BLOCK_LINEARITY; }

      const char* GetName() const
      { return "<blocklinearity>"; }

      Component* Clone() const 
      { 
        BlockLinearity* p = new BlockLinearity(GetNInputs(),GetNOutputs(),NULL);
        p->mBlockLinearity.Init(mBlockLinearity.Rows(),mBlockLinearity.Cols()); 
        p->mBlockLinearity.Copy(mBlockLinearity); 
        return p; 
      }

      void PropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y) 
      {
        assert(X.Rows() == Y.Rows());
        assert(X.Cols()%mBlockLinearity.Rows() == 0);
        assert(Y.Cols()%mBlockLinearity.Cols() == 0);
        assert(X.Cols()/mBlockLinearity.Rows() == Y.Cols()/mBlockLinearity.Cols());
        
        int instN = X.Cols()/mBlockLinearity.Rows();
        for(int inst=0; inst<instN; inst++) {
#ifndef DOUBLEPRECISION
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  X.Rows(), mBlockLinearity.Cols(), mBlockLinearity.Rows(),
                  1.0, X.pData()+inst*mBlockLinearity.Rows(), X.Stride(), 
                  mBlockLinearity.pData(), mBlockLinearity.Stride(),
                  0.0, Y.pData()+inst*mBlockLinearity.Cols(), Y.Stride());
#else
          cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  X.Rows(), mBlockLinearity.Cols(), mBlockLinearity.Rows(),
                  1.0, X.pData()+inst*mBlockLinearity.Rows(), X.Stride(), 
                  mBlockLinearity.pData(), mBlockLinearity.Stride(),
                  0.0, Y.pData()+inst*mBlockLinearity.Cols(), Y.Stride());
#endif
        }
      }
        
      void BackpropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y) 
      { Error("__func__ Not implemented"); }


      void ReadFromStream(std::istream& rIn)
      { 
        Matrix<BaseFloat> mat;
        rIn >> mat;
        Matrix<BaseFloat> trans(mat,TRANS);
        mBlockLinearity.Init(trans.Rows(),trans.Cols()).Copy(trans);

        if((GetNOutputs() % mBlockLinearity.Cols() != 0) ||
           (GetNInputs() % mBlockLinearity.Rows() != 0) ||
           ((GetNOutputs() / mBlockLinearity.Cols()) != 
            (GetNInputs() / mBlockLinearity.Rows()))) 
        {
          Error("BlockLinearity matrix dimensions must divide IO dims");
        }
      }

      void WriteToStream(std::ostream& rOut)
      {
        Matrix<BaseFloat> trans(mBlockLinearity,TRANS);
        rOut << trans;
      }

    private:
      Matrix<BaseFloat> mBlockLinearity;
  };


  
  class Bias : public Component
  {
    public:
      Bias(size_t nInputs, size_t nOutputs, Component* pPred)
        : Component(nInputs,nOutputs,pPred)
      { }

      ~Bias()
      { }


      ComponentType GetType() const
      { return Component::BIAS; }

      const char* GetName() const
      { return "<bias>"; }

      Component* Clone() const  
      { 
        Bias* p = new Bias(GetNInputs(),GetNOutputs(),NULL);
        p->mBias.Init(mBias.Dim()); 
        p->mBias.Copy(mBias); 
        return p; 
      }

      void PropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y)
      { 
        Y.Copy(X); 
        for(size_t r=0; r<X.Rows(); r++) {
          for(size_t c=0; c<X.Cols(); c++) {
            Y(r,c) += mBias[c];
          }
        }
      }

      void BackpropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y)
      { Y.Copy(X); }
  
     
      void ReadFromStream(std::istream& rIn)
      { rIn >> mBias; }

      void WriteToStream(std::ostream& rOut)
      { rOut << mBias; }

    private:
      Vector<BaseFloat> mBias;
  };



  class Window : public Component
  {
    public:
      Window(size_t nInputs, size_t nOutputs, Component* pPred)
        : Component(nInputs, nOutputs, pPred)
      { }

      ~Window()
      { }


      ComponentType GetType() const
      { return Component::WINDOW; }

      const char* GetName() const
      { return "<window>"; }

      Component* Clone() const  
      { 
        Window* p = new Window(GetNInputs(),GetNOutputs(),NULL);
        p->mWindow.Init(mWindow.Dim()); 
        p->mWindow.Copy(mWindow); 
        return p; 
      }


      void PropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y)
      { Y.Copy(X); 
        for(size_t r=0; r<X.Rows(); r++) {
          for(size_t c=0; c<X.Cols(); c++) {
            Y(r,c) *= mWindow[c];
          }
        }
      }

      void BackpropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y)
      { Error("__func__ Not implemented"); }
     
      
      void ReadFromStream(std::istream& rIn)
      { rIn >> mWindow; }

      void WriteToStream(std::ostream& rOut)
      { rOut << mWindow; }

    private:
      Vector<BaseFloat> mWindow;
  };

  class Log : public Component
  {
    public:
      Log(size_t nInputs, size_t nOutputs, Component* pPred)
        : Component(nInputs, nOutputs, pPred)
      { }

      ~Log()
      { }


      ComponentType GetType() const
      { return Component::LOG; }

      const char* GetName() const
      { return "<log>"; }

      Component* Clone() const  
      { return new Log(GetNInputs(),GetNOutputs(),NULL); }


      void PropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y)
      { Y.Copy(X); Y.ApplyLog(); }

      void BackpropagateFnc(const Matrix<BaseFloat>& X, Matrix<BaseFloat>& Y)
      { Error("__func__ Not implemented"); }
     
      
      void ReadFromStream(std::istream& rIn)
      { }

      void WriteToStream(std::ostream& rOut)
      { }

  };

}


#endif

