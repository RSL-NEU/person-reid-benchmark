/*
 * =============================================================
 * sd.c 
  
 * equivalent to running sd2b.m
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
  int Ng0,ig0;
  double *g0;

  int Ng1,Na1,ia1,iplus1,iminus1,iactive1;
  double *g1, *a1 , *plus1, *minus1,*active1;

  int Ng2,Na2,ia2,iplus2,iminus2,iactive2;
  double *g2, *a2 , *plus2, *minus2,*active2;


  /* Check for proper number of input and output arguments. */    
  if (nrhs != 5) {
    mexErrMsgTxt("Three input arguments required.");
  } 
  if (nlhs > 6) {    mexErrMsgTxt("Too many output arguments.");}

  /* Check data type of input argument. */
  if (!(mxIsDouble(prhs[0]))) {
    mexErrMsgTxt("Input array must be of type double.");
  }
    
  /* Get the number of elements in the input argument. */
  Ng0 = mxGetNumberOfElements(prhs[0]);
  Ng1 = mxGetNumberOfElements(prhs[1]);
  Na1 = mxGetNumberOfElements(prhs[2]);  
  Ng2 = mxGetNumberOfElements(prhs[3]);
  Na2 = mxGetNumberOfElements(prhs[4]);  

  if (Ng0!=Ng1) {    mexErrMsgTxt("First two vectors must have equal length.");}
  if (Ng0!=Ng2) {    mexErrMsgTxt("First and third vectors must have equal length.");}
  
  /* Get the data. */
  g0  = (double *)mxGetPr(prhs[0]);
  g1  = (double *)mxGetPr(prhs[1]);
  a1  = (double *)mxGetPr(prhs[2]);
  g2  = (double *)mxGetPr(prhs[3]);
  a2  = (double *)mxGetPr(prhs[4]);

  
  plhs[0]=mxCreateDoubleMatrix(1,Ng0,mxREAL);  plus1=mxGetPr(plhs[0]);
  plhs[1]=mxCreateDoubleMatrix(1,Na1,mxREAL);  minus1=mxGetPr(plhs[1]);
  plhs[2]=mxCreateDoubleMatrix(1,Ng0,mxREAL);  active1=mxGetPr(plhs[2]);
  plhs[3]=mxCreateDoubleMatrix(1,Ng0,mxREAL);  plus2=mxGetPr(plhs[3]);
  plhs[4]=mxCreateDoubleMatrix(1,Na2,mxREAL);  minus2=mxGetPr(plhs[4]);
  plhs[5]=mxCreateDoubleMatrix(1,Ng0,mxREAL);  active2=mxGetPr(plhs[5]);

  ig0=0;
  iplus1=0;iminus1=0;ia1=0;iactive1=0;
  iplus2=0;iminus2=0;ia2=0;iactive2=0;
  while(ig0<Ng0){
    if (g0[ig0]<g1[ig0]){
     active1[iactive1]=ig0+1;
     iactive1++; /* count this as active1 constraint*/
     while(ia1<Na1 && (ig0+1>a1[ia1]))  {
         minus1[iminus1]=a1[ia1];
         iminus1++;
         ia1++;
     }     
     plus1[iplus1]=ig0+1;
      if(ia1<Na1 && (ig0+1==a1[ia1])) ia1++;
      else iplus1++;     
    }  

    if (g0[ig0]<g2[ig0]){
     active2[iactive2]=ig0+1;
     iactive2++; /* count this as active2 constraint*/
     while(ia2<Na2 && (ig0+1>a2[ia2]))  {
         minus2[iminus2]=a2[ia2];
         iminus2++;
         ia2++;
     }     
     plus2[iplus2]=ig0+1;
      if(ia2<Na2 && (ig0+1==a2[ia2])) ia2++;
      else iplus2++;     
    }  
    ig0++;
  }
  for(;ia1<Na1;ia1=ia1+1) {minus1[iminus1]=a1[ia1];iminus1=iminus1+1;}
  for(;ia2<Na2;ia2=ia2+1) {minus2[iminus2]=a2[ia2];iminus2=iminus2+1;}
   
  mxSetN(plhs[0],iplus1);
  mxSetN(plhs[1],iminus1);
  mxSetN(plhs[2],iactive1);  
  mxSetN(plhs[3],iplus2);
  mxSetN(plhs[4],iminus2);
  mxSetN(plhs[5],iactive2);  
}

