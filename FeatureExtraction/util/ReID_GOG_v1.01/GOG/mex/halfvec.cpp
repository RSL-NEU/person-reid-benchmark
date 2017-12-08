// vec = halfvec( X  )
// half vectorize symmetric matrix around I 
// 

#include "mex.h"
#include <string.h>

#ifdef INT_2_BYTES
    typedef char      int8;
    typedef int       int16;
    typedef long      int32;
    typedef long long int64;
    
    typedef unsigned char      uint8;
    typedef unsigned int       uint16;
    typedef unsigned long      uint32;
    typedef unsigned long long uint64;
#else
    typedef char      int8;
    typedef short     int16;
    typedef int       int32;
    typedef long long int64;
    
    typedef unsigned char      uint8;
    typedef unsigned short     uint16;
    typedef unsigned int       uint32;
    typedef unsigned long long uint64;
#endif

static const int ARG_NUM_X = 0;
static const int ARG_NUM_OUT = 0;

//-------------------------------------------------------------------------
template < typename dataType, mxClassID mxClassId  >
inline void halfvec(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])
{
  /* Declare variables. */ 

  dataType *X, *OUT;
  int m,n;
  int outdim; 
  int ind, c, r;
     
  /* Get the input argument. */
  n = mxGetN(prhs[ARG_NUM_X]);
  m = mxGetM(prhs[ARG_NUM_X]);

  outdim = n*(n+1)/2;
  
  /* Get the data. */
  X   = (dataType*)mxGetPr(prhs[ARG_NUM_X]);
  
  /* Create output matrix */
  plhs[ARG_NUM_OUT]=mxCreateNumericMatrix(outdim,1,mxClassId,mxREAL);
  OUT =(dataType*)mxGetPr(plhs[ARG_NUM_OUT]);
  memset(OUT,0,sizeof(dataType)*outdim);
   
  /*
  ind = 0;
   for(r=0;r<m;r++)
	 for(c=r+1;c<m;c++){
     OUT[ind]= 1.4142*X[r+c*m];
     ind = ind + 1;
     }
  for( r=0; r<m; r++){
      OUT[ind] = X[r+r*m];
      ind = ind + 1;
  }
  */
   ind = 0;
   for(r=0;r<m;r++){
      OUT[ind] = X[r+r*m];
      ind = ind + 1;
	  for(c=r+1;c<m;c++){
          OUT[ind]= 1.4142*X[r+c*m];
          ind = ind + 1;
      }
   } 
 }

//-------------------------------------------------------------------------
void mexFunction(int outputSize, mxArray *output[], int inputSize, const mxArray *input[]) 
{
  /* Check for proper number of input and output arguments. */    
  if (inputSize != 1) {
    mexErrMsgTxt("Exactly one input arguments required.");
  } 

  if (outputSize > 1) {
    mexErrMsgTxt("Too many output arguments.");
  }
  
   /* Check data type of input arguments is the same. */
 // if (!(mxGetClassID(input[ARG_NUM_X]) == mxGetClassID(input[ARG_NUM_IDXA]) && mxGetClassID(input[ARG_NUM_X]) == mxGetClassID(input[ARG_NUM_IDXB]))) {
 //  mexErrMsgTxt("inputs should be of same type.");
 // }
  
  try
  {
    switch(mxGetClassID(input[ARG_NUM_X]))
    {
        case mxUINT16_CLASS:
            if(sizeof(uint16)!=2)
                mexErrMsgTxt("error uint16");
            halfvec<uint16,mxUINT16_CLASS>(outputSize, output, inputSize, input);        
            break;
        case mxINT16_CLASS:
            if(sizeof(int16)!=2)
                mexErrMsgTxt("error int16");
            halfvec<int16,mxINT16_CLASS>(outputSize, output, inputSize, input);              
            break;
        case mxUINT32_CLASS:
            if(sizeof(uint32)!=4)
                mexErrMsgTxt("error uint32");
            halfvec<uint32,mxUINT32_CLASS>(outputSize, output, inputSize, input);        
            break;
        case mxINT32_CLASS:
            if(sizeof(int32)!=4)
                mexErrMsgTxt("error int32");
            halfvec<int32,mxINT32_CLASS>(outputSize, output, inputSize, input);           
            break;
        case mxUINT64_CLASS:
            if(sizeof(uint64)!=8)
                mexErrMsgTxt("error uint64");
            halfvec<uint64,mxUINT64_CLASS>(outputSize, output, inputSize, input);            
            break;
        case mxINT64_CLASS:
            if(sizeof(int64)!=8)
                mexErrMsgTxt("error int64");
            halfvec<int64,mxINT64_CLASS>(outputSize, output, inputSize, input);            
            break;
		case mxSINGLE_CLASS:
            halfvec<float,mxSINGLE_CLASS>(outputSize, output, inputSize, input);          
            break;	
        case mxDOUBLE_CLASS:
            halfvec<double,mxDOUBLE_CLASS>(outputSize, output, inputSize, input);          
            break;
        default:
           mexErrMsgTxt("sorry class not supported!");         
    }
  }
  catch(...)
  {
    mexErrMsgTxt("Internal error");
  }
}
//-------------------------------------------------------------------------



