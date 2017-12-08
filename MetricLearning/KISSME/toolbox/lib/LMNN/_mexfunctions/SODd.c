/*
 * =============================================================
 * SODd.c 
  
 * input: x,a,b
 *    x : matrix DxN
 *    a : vector 1xnn
 *    b : vector 1xnn
 *
 * output: for i=1:nn; res=res+x(:,a(i))*x(:,b(i))';end;
 * 
 * =============================================================
 */

/* $Revision: 1.1 $ */

#include "mex.h"

/* If you are using a compiler that equates NaN to zero, you must
 * compile this example using the flag -DNAN_EQUALS_ZERO. For 
 * example:
 *
 *     mex -DNAN_EQUALS_ZERO findnz.c  
 *
 * This will correctly define the IsNonZero macro for your
   compiler. */

#if NAN_EQUALS_ZERO
#define IsNonZero(d) ((d) != 0.0 || mxIsNaN(d))
#else
#define IsNonZero(d) ((d) != 0.0)
#endif


double square(double x) { return(x*x);}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  /* Declare variables. */ 

  double *X, *v1,*v2, *C;
  double *av,*bv;
  int m,p,n,inds;
  int j,i,r,c;
  int ione=1;
  char *chu="U"; 
  char *chl="L";
  char *chn2="T";
  char *chn="N";
  double dummy,minusone=-1.0,one=1.0, zero=0.0;



  /* Check for proper number of input and output arguments. */    
  if (nrhs != 3) {
    mexErrMsgTxt("Exactly three input arguments required.");
  } 

  if (nlhs > 1) {
    mexErrMsgTxt("Too many output arguments.");
  }

  /* Check data type of input argument. */
  if (!(mxIsDouble(prhs[0]))) {
   mexErrMsgTxt("Input array must be of type double.");
  }
  /* Check data type of input argument. */
  /*  if ((mxIsDouble(prhs[1]))) {
   mexErrMsgTxt("Input array must be of type double.");
  }
  if ((mxIsDouble(prhs[2]))) {
   mexErrMsgTxt("Input array must be of type double.");
   }*/

      
  /* Get the number of elements in the input argument. */
  inds = mxGetNumberOfElements(prhs[1]);
  if(inds != mxGetNumberOfElements(prhs[2]))
    mxErrMsgTxt("Hey Bongo! Both index vectors must have same length!\n");
  n = mxGetN(prhs[0]);
  m = mxGetM(prhs[0]);

  /* Get the data. */
  X  = mxGetPr(prhs[0]);
  av  = mxGetPr(prhs[1]);
  bv  = mxGetPr(prhs[2]);


  /* Create output matrix */
  plhs[0]=mxCreateDoubleMatrix(m,m,mxREAL);
  C=mxGetPr(plhs[0]);

  for(i=0;i<inds;i++){
   /* Assign cols addresses */
   v1=&X[(int) (av[i]-1)*m];
   v2=&X[(int) (bv[i]-1)*m];

/*   dsyr_(chu,&m,&one,v1,&ione,C,&m);
   dsyr_(chu,&m,&one,v2,&ione,C,&m);
   dsyr2_(chu,&m,&minusone,v1,&ione,v2,&ione,C,&m);*/
   j=0;
   for(c=0;c<m;c++){
	 dummy=v1[c]-v2[c];	
	 C[j]+=dummy*dummy;
	 j+=m+1;
   }    
  }

      /*  sx2=mxCalloc(n,sizeof(double));*/
}



