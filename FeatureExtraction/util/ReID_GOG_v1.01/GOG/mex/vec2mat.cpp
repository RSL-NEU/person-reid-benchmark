// S = vec2mat( x, d  )
// convert d*(d-1)/2 dim vector into d*d dim symmetric matrix S

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
static const int ARG_NUM_D = 1;
static const int ARG_NUM_OUT = 0;

//-------------------------------------------------------------------------
template < typename dataType, mxClassID mxClassId  >
inline void vec2mat(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])
{
  /* Declare variables. */ 

  dataType *X, *d, *OUT;
 
  int ind, c, r;
     
  /* Get the data. */
  X   = (dataType*)mxGetPr(prhs[ARG_NUM_X]);
  d   = (dataType*)mxGetPr(prhs[ARG_NUM_D]);
  int dim = (int)d[0];
  
  const int *dim_array;
  dim_array = mxGetDimensions( prhs[ARG_NUM_X] ); 
  int h = dim_array[0];
  int w = dim_array[1];
  
  /* Create output matrix */
  plhs[ARG_NUM_OUT]=mxCreateNumericMatrix(dim,dim,mxClassId,mxREAL);
  OUT =(dataType*)mxGetPr(plhs[ARG_NUM_OUT]);
  memset(OUT,0,sizeof(dataType)*dim*dim);
   
  ind = 0;
   for(r=0;r<dim;r++)
	 for(c=r;c<dim;c++){
     OUT[r+c*dim] = X[ind];
     OUT[c+r*dim] = X[ind];
     ind = ind + 1;
     }
   
  if(h  != ind && w == 1 || w != ind && h == 1){
      mexPrintf(" Error ALL Dimension should be used for vec2Mat ! \n");
     // while(1);
  }
  
 }

//-------------------------------------------------------------------------
void mexFunction(int outputSize, mxArray *output[], int inputSize, const mxArray *input[]) 
{
  /* Check for proper number of input and output arguments. */    
  if (inputSize != 2) {
    mexErrMsgTxt("Exactly two input arguments required.");
  } 

  if (outputSize > 2) {
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
            vec2mat<uint16,mxUINT16_CLASS>(outputSize, output, inputSize, input);        
            break;
        case mxINT16_CLASS:
            if(sizeof(int16)!=2)
                mexErrMsgTxt("error int16");
            vec2mat<int16,mxINT16_CLASS>(outputSize, output, inputSize, input);              
            break;
        case mxUINT32_CLASS:
            if(sizeof(uint32)!=4)
                mexErrMsgTxt("error uint32");
            vec2mat<uint32,mxUINT32_CLASS>(outputSize, output, inputSize, input);        
            break;
        case mxINT32_CLASS:
            if(sizeof(int32)!=4)
                mexErrMsgTxt("error int32");
            vec2mat<int32,mxINT32_CLASS>(outputSize, output, inputSize, input);           
            break;
        case mxUINT64_CLASS:
            if(sizeof(uint64)!=8)
                mexErrMsgTxt("error uint64");
            vec2mat<uint64,mxUINT64_CLASS>(outputSize, output, inputSize, input);            
            break;
        case mxINT64_CLASS:
            if(sizeof(int64)!=8)
                mexErrMsgTxt("error int64");
            vec2mat<int64,mxINT64_CLASS>(outputSize, output, inputSize, input);            
            break;
		case mxSINGLE_CLASS:
            vec2mat<float,mxSINGLE_CLASS>(outputSize, output, inputSize, input);          
            break;	
        case mxDOUBLE_CLASS:
            vec2mat<double,mxDOUBLE_CLASS>(outputSize, output, inputSize, input);          
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



