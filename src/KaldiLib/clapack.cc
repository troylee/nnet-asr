
extern "C" {

  /**
   * Wrapper to GotoBLAS lapack for STK and TNet (sgetrf sgetri dgetrf dgetri)
   */
  typedef float real;
  typedef double doublereal;
  typedef int integer;


  /**
   * The lapack interface (used in gotoblas)
   */
  /* Subroutine */ int sgetrf_(integer *m, integer *n, real *a, integer *lda,
          integer *ipiv, integer *info);
  /* Subroutine */ int sgetri_(integer *n, real *a, integer *lda, integer *ipiv,
          real *work, integer *lwork, integer *info);
  /* Subroutine */ int dgetrf_(integer *m, integer *n, doublereal *a, integer *
          lda, integer *ipiv, integer *info);
  /* Subroutine */ int dgetri_(integer *n, doublereal *a, integer *lda, integer
          *ipiv, doublereal *work, integer *lwork, integer *info);





  /**
   * The clapack interface as used by ATLAS (used in STK, 
   */
  enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102 };

  int clapack_sgetrf(const enum CBLAS_ORDER Order, const int M, const int N,
                     float *A, const int lda, int *ipiv) 
  {
    return sgetrf_((int*)&M, (int*)&N, A, (int*)&lda, (int*)ipiv, 0);
  }


  int clapack_sgetri(const enum CBLAS_ORDER Order, const int N, float *A,
                     const int lda, const int *ipiv) 
  {
    return sgetri_((int*)&N, A, (int*)&lda, (int*)ipiv, 0, 0, 0);
  }


  int clapack_dgetrf(const enum CBLAS_ORDER Order, const int M, const int N,
                     double *A, const int lda, int *ipiv) 
  {
    return dgetrf_((int*)&M, (int*)&N, A, (int*)&lda, (int*)ipiv, 0);
  }
    

  int clapack_dgetri(const enum CBLAS_ORDER Order, const int N, double *A,
                     const int lda, const int *ipiv)
  {
    return dgetri_((int*)&N, A, (int*)&lda, (int*)ipiv, 0, 0, 0);
  }


}
