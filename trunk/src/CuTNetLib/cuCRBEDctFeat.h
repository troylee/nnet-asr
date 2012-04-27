#ifndef _CUCRBEDCTFEATURES_H_
#define _CUCRBEDCTFEATURES_H_


#include "cuComponent.h"
#include "cumath.h"


namespace TNet {

  /**
   * Expands the time context of the input features
   * in N, out k*N, FrameOffset o_1,o_2,...,o_k
   * FrameOffset example 11frames: -5 -4 -3 -2 -1 0 1 2 3 4 5
   */
  class CuExpand : public CuComponent
  {
   public:
    CuExpand(size_t nInputs, size_t nOutputs, CuComponent* pPred)
      : CuComponent(nInputs,nOutputs,pPred)
    { }

    ~CuExpand()
    { }

    ComponentType GetType() const
    { return EXPAND; }

    const char* GetName() const
    { return "<expand>"; }
   
    void ReadFromStream(std::istream& rIn)
    { Vector<int> vec; rIn >> vec; mFrameOffset.CopyFrom(vec); }

    void WriteToStream(std::ostream& rOut)  
    { Vector<int> vec; mFrameOffset.CopyTo(vec); rOut << vec; }
     
   protected:
    void PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
    { CuMath<BaseFloat>::Expand(Y,X,mFrameOffset); }

    void BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
    { Error("__func__ Nonsense"); }

   protected:
    CuVector<int> mFrameOffset;
  };



  /**
   * Rearrange the matrix columns according to the indices in mCopyFromIndices
   */
  class CuCopy : public CuComponent
  {
   public:
    CuCopy(size_t nInputs, size_t nOutputs, CuComponent* pPred)
      : CuComponent(nInputs,nOutputs,pPred)
    { }

    ~CuCopy()
    { }

    ComponentType GetType() const
    { return COPY; }

    const char* GetName() const
    { return "<copy>"; }
   
    void ReadFromStream(std::istream& rIn)
    { Vector<int> vec; rIn >> vec; vec.Add(-1); mCopyFromIndices.CopyFrom(vec); }

    void WriteToStream(std::ostream& rOut)  
    { Vector<int> vec; mCopyFromIndices.CopyTo(vec); vec.Add(1); rOut << vec; }
     
   protected:
    void PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
    { CuMath<BaseFloat>::Rearrange(Y,X,mCopyFromIndices); }

    void BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
    { Error("__func__ Nonsense"); }

   protected:
    CuVector<int> mCopyFromIndices;
  };
  
  class CuTranspose : public CuComponent
  {
   public:
    CuTranspose(size_t nInputs, size_t nOutputs, CuComponent* pPred)
      : CuComponent(nInputs,nOutputs,pPred), mContext(0)
    { }

    ~CuTranspose()
    { }

    ComponentType GetType() const
    { return TRANSPOSE; }

    const char* GetName() const
    { return "<transpose>"; }
   
    void ReadFromStream(std::istream& rIn)
    { 
      rIn >> std::ws >> mContext;

      if(GetNInputs() != GetNOutputs()) { 
        Error("Input dim must be same as output dim"); 
      }
      if(GetNInputs() % mContext != 0) { 
        Error("Number of inputs must be divisible by context length"); 
      }

      Vector<int> vec(GetNInputs());
      int channels = GetNInputs() / mContext;
      for(int i=0, ch=0; ch<channels; ch++) {
        for(int idx=ch; idx < (int)GetNInputs(); idx+=channels, i++) {
          assert(i < (int)GetNInputs());
          vec[i] = idx;
        }
      }

      mCopyFromIndices.CopyFrom(vec); 
    }

    void WriteToStream(std::ostream& rOut)  
    { rOut << " " << mContext << "\n"; }
     
   protected:
    void PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
    { CuMath<BaseFloat>::Rearrange(Y,X,mCopyFromIndices); }

    void BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
    { Error("__func__ Nonsense"); }

   protected:
    int mContext;
    CuVector<int> mCopyFromIndices;
  };


  /**
   * CuBlockLinearity is used for the blockwise multiplication by 
   * DCT transform loaded from disk
   */
  class CuBlockLinearity : public CuComponent
  {
    public:
      CuBlockLinearity(size_t nInputs, size_t nOutputs, CuComponent* pPred)
        : CuComponent(nInputs,nOutputs,pPred)
      { }

      ~CuBlockLinearity()
      { }


      ComponentType GetType() const
      { return CuComponent::BLOCK_LINEARITY; }

      const char* GetName() const
      { return "<blocklinearity>"; }


      void PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y) 
      { CuMath<BaseFloat>::BlockLinearity(Y,X,mBlockLinearity); }
        
      void BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y) 
      { Error("__func__ Not implemented"); }


      void ReadFromStream(std::istream& rIn)
      { 
        Matrix<BaseFloat> mat;
        rIn >> mat;
        Matrix<BaseFloat> trans(mat,TRANS);
        mBlockLinearity.CopyFrom(trans);

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
        Matrix<BaseFloat> mat;
        mBlockLinearity.CopyTo(mat);
        Matrix<BaseFloat> trans(mat,TRANS);
        rOut << trans;
      }

    private:
      CuMatrix<BaseFloat> mBlockLinearity;
  };


  
  class CuBias : public CuComponent
  {
    public:
      CuBias(size_t nInputs, size_t nOutputs, CuComponent* pPred)
        : CuComponent(nInputs,nOutputs,pPred)
      { }

      ~CuBias()
      { }


      ComponentType GetType() const
      { return CuComponent::BIAS; }

      const char* GetName() const
      { return "<bias>"; }


      void PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
      { Y.CopyFrom(X); Y.AddScaledRow(1.0, mBias, 1.0); }

      void BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
      { Y.CopyFrom(X); }
  
     
      void ReadFromStream(std::istream& rIn)
      { Vector<BaseFloat> vec; rIn >> vec; mBias.CopyFrom(vec); }

      void WriteToStream(std::ostream& rOut)
      { Vector<BaseFloat> vec; mBias.CopyTo(vec); rOut << vec; }

    private:
      CuVector<BaseFloat> mBias;
  };



  class CuWindow : public CuComponent
  {
    public:
      CuWindow(size_t nInputs, size_t nOutputs, CuComponent* pPred)
        : CuComponent(nInputs, nOutputs, pPred)
      { }

      ~CuWindow()
      { }


      ComponentType GetType() const
      { return CuComponent::WINDOW; }

      const char* GetName() const
      { return "<window>"; }


      void PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
      { Y.CopyFrom(X); Y.ScaleCols(mWindow); }

      void BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
      { Error("__func__ Not implemented"); }
     
      
      void ReadFromStream(std::istream& rIn)
      { Vector<BaseFloat> vec; rIn >> vec; mWindow.CopyFrom(vec); }

      void WriteToStream(std::ostream& rOut)
      { Vector<BaseFloat> vec; mWindow.CopyTo(vec); rOut << vec; }

    private:
      CuVector<BaseFloat> mWindow;
  };

  class CuLog : public CuComponent
  {
    public:
      CuLog(size_t nInputs, size_t nOutputs, CuComponent* pPred)
        : CuComponent(nInputs, nOutputs, pPred)
      { }

      ~CuLog()
      { }


      ComponentType GetType() const
      { return CuComponent::LOG; }

      const char* GetName() const
      { return "<log>"; }


      void PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
      { Y.CopyFrom(X); Y.ApplyLog(); }

      void BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
      { Error("__func__ Not implemented"); }
     
      
      void ReadFromStream(std::istream& rIn)
      { }

      void WriteToStream(std::ostream& rOut)
      { }

  };

}


#endif

