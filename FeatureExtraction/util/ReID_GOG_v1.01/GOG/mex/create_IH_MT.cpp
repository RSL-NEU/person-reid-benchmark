// IH = create_IH_MT( X )
// create Integral Histogram with MultiThread
// size(X) = [h, w, channel]
// X must be double

#include "mex.h"
#include <string.h>

#include <stdio.h>
#include <windows.h>
#include <process.h>    

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

#define threadnum 4

double *X, *IH;
int h, w, numchannel;
int h2, w2;

typedef struct{
    int thid;
}PARAM, *lpPARAM;

unsigned creIH(void *lpx){
    
    lpPARAM lpParam = (lpPARAM) lpx;
    
    int tid = lpParam->thid;
    int startc, incc;
    startc = lpParam->thid -1;
    incc = threadnum;
    
    for(int ch = startc; ch< numchannel; ch = ch + incc){
      // x = 0, y =0
      IH[ 1 + h2 + ch*w2*h2] = X[ch*w*h];
      
      // y = 0
      for(int x = 1; x<w; x++)
          IH[ 1+ (x+1)*h2 + ch*w2*h2] = IH[ 1 + x*h2 + ch*w2*h2] + X[  x*h + ch*w*h]; 
     
      for(int y = 1; y<h; y++){ 
          IH[ y+1 + h2 +  ch*w2*h2] = IH[ y + h2 + ch*w2*h2] +  X[y + ch*w*h]; //x = 0 
          for(int x = 1; x <w; x++) 
              IH[ y+1 + (x+1)*h2 + ch*w2*h2] = IH[ y+1 + x*h2 + ch*w2*h2 ] +  IH[ y + (x+1)*h2 + ch*w2*h2] +  X[y + x*h + ch*w*h] - IH[ y + x*h2 + ch*w2*h2];
      }
   }
        
    return 0;
}


//-------------------------------------------------------------------------
template < typename dataType, mxClassID mxClassId  >
inline void CreateIH(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])
{
  /* Declare variables. */
  //dataType *X, *IH;
  
  // Get the input argument
  X   = (dataType*)mxGetPr(prhs[ARG_NUM_X]);
  
  int numdim = mxGetNumberOfDimensions(prhs[ARG_NUM_X]);
  if ( numdim < 2 )
        mexErrMsgTxt("number of dimensions of input must be at least 2 !\n");
  
  
  
  const int *dim_array;
  dim_array = mxGetDimensions( prhs[ARG_NUM_X] ); 
  h = dim_array[0];
  w = dim_array[1];
 
  if ( numdim == 2 ){
      numchannel = 1;
  }
  else{
      numchannel = dim_array[2];
  }
  
  /* Create output matrix */
  int *dim_array_out;
  dim_array_out = (int*)malloc(sizeof(int)*numdim );
  h2 = h + 1;
  w2 = w + 1;
  dim_array_out[0] = h2;
  dim_array_out[1] = w2;
  if(numdim >= 2) dim_array_out[2] = numchannel;
  
  plhs[ARG_NUM_OUT]=mxCreateNumericArray( numdim, dim_array_out, mxClassId, mxREAL);
  IH = (dataType*)mxGetPr(plhs[ARG_NUM_OUT]);
  
  memset(IH,0,sizeof(dataType)*h2*w2*numchannel);
  
  //multi thread Ç…ÇÊÇÈèàóù
  DWORD thID[threadnum];
  HANDLE hThreads[threadnum];
  PARAM param[threadnum];
  for (int t = 0; t< threadnum; t= t + 1){
      param[t].thid = t + 1;
      //param[t].Data = X;
  }
  
  for( int t= 0; t< threadnum; t = t + 1)
    hThreads[t] = (HANDLE) _beginthreadex( NULL, 0, creIH, &param[t], 0, (unsigned int*)&thID[t]);
  
  WaitForMultipleObjects(threadnum , hThreads, TRUE, INFINITE);	// wait for both threads to terminate
   
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
        /*
        case mxUINT16_CLASS:
            if(sizeof(uint16)!=2)
                mexErrMsgTxt("error uint16");
            createcovvec<uint16,mxUINT16_CLASS>(outputSize, output, inputSize, input);        
            break;
        case mxINT16_CLASS:
            if(sizeof(int16)!=2)
                mexErrMsgTxt("error int16");
            createcovvec<int16,mxINT16_CLASS>(outputSize, output, inputSize, input);              
            break;
        case mxUINT32_CLASS:
            if(sizeof(uint32)!=4)
                mexErrMsgTxt("error uint32");
            createcovvec<uint32,mxUINT32_CLASS>(outputSize, output, inputSize, input);        
            break;
        case mxINT32_CLASS:
            if(sizeof(int32)!=4)
                mexErrMsgTxt("error int32");
            createcovvec<int32,mxINT32_CLASS>(outputSize, output, inputSize, input);           
            break;
        case mxUINT64_CLASS:
            if(sizeof(uint64)!=8)
                mexErrMsgTxt("error uint64");
            createcovvec<uint64,mxUINT64_CLASS>(outputSize, output, inputSize, input);            
            break;
        case mxINT64_CLASS:
            if(sizeof(int64)!=8)
                mexErrMsgTxt("error int64");
            createcovvec<int64,mxINT64_CLASS>(outputSize, output, inputSize, input);            
            break;
		case mxSINGLE_CLASS:
            createcovvec<float,mxSINGLE_CLASS>(outputSize, output, inputSize, input);          
            break;
         */	
        case mxDOUBLE_CLASS:
            CreateIH<double,mxDOUBLE_CLASS>(outputSize, output, inputSize, input);          
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



