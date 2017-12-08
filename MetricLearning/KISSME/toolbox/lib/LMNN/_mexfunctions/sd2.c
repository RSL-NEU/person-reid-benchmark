/*
 * =============================================================
 * sd.c 
  
 * takes two input sorted input vectors and finds the difference
 * 
 * =============================================================
 */

/* $Revision: 1.2 $ */

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



void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  /* Declare variables. */ 
  int N1,N2,N3, o1,o2,i1,i3,o3;
  int size1,size2;
  double *pi1, *pi2, *pi3 , *po1, *po2, *active;
  int mRows,nCols;



  /* Check for proper number of input and output arguments. */    
  if (nrhs != 3) {
    mexErrMsgTxt("Three input arguments required.");
  } 
  if (nlhs > 3) {    mexErrMsgTxt("Too many output arguments.");}

  /* Check data type of input argument. */
  if (!(mxIsDouble(prhs[0]))) {
    mexErrMsgTxt("Input array must be of type double.");
  }
    
  /* Get the number of elements in the input argument. */
  N1 = mxGetNumberOfElements(prhs[0]);
  N2 = mxGetNumberOfElements(prhs[1]);
  N3 = mxGetNumberOfElements(prhs[2]);  

  if (N1!=N2) {    mexErrMsgTxt("First two vectors must have equal length.");}
  

  /* Get the data. */
  pi1  = (double *)mxGetPr(prhs[0]);
  pi2  = (double *)mxGetPr(prhs[1]);
  pi3  = (double *)mxGetPr(prhs[2]);

  
  plhs[0]=mxCreateDoubleMatrix(1,N1,mxREAL);
  po1=mxGetPr(plhs[0]);
  plhs[1]=mxCreateDoubleMatrix(1,N3,mxREAL);
  po2=mxGetPr(plhs[1]);
  plhs[2]=mxCreateDoubleMatrix(1,N1,mxREAL);  
  active=mxGetPr(plhs[2]);

  i1=0;i3=0;
  o1=0;o2=0;o3=0;
  while(i1<N1){
    active[o3]=i1+1;
    if (pi1[i1]<pi2[i1]){
     o3++; /* count this as active constraint*/
     while(i3<N3 && (i1+1>pi3[i3]))  {
         po2[o2]=pi3[i3];
         o2++;
         i3++;
     }     
     po1[o1]=i1+1;
      if(i3<N3 && (i1+1==pi3[i3])) i3++;
      else o1++;     
    }  
    i1++;
  }
  for(;i3<N3;i3=i3+1) {po2[o2]=pi3[i3];o2=o2+1;}
   
  mxSetN(plhs[0],o1);
  mxSetN(plhs[1],o2);
  mxSetN(plhs[2],o3);  
}

