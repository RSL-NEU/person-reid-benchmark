#include <math.h>
#include "mex.h"

/* Input Arguments */
#define	X				prhs[0]
#define	MASK			prhs[1]
#define MASK_SCALE		prhs[2]
#define UNMASK_SCALE	prhs[3]

/* Output Arguments */
#define	Y				plhs[0]

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	mwSize nsample;
	bool *mask;
	double *y, mask_scale, unmask_scale;
	float *f_y, f_mask_scale, f_unmask_scale;

	/* Check for proper number of arguments */
	if (nrhs != 4)
		mexErrMsgTxt("4 input arguments required.");
	else if (nlhs > 1)
		mexErrMsgTxt("Too many output arguments.");

	/* Check input arguments */
	if (!mxIsDouble(X) && !mxIsSingle(X))
		mexErrMsgTxt("Input data matrix must be double or single.");
	if (!mxIsLogical(MASK))
		mexErrMsgTxt("Input mask matrix must be logical.");
	if (mxGetNumberOfElements(X) != mxGetNumberOfElements(MASK))
		mexErrMsgTxt("Input data and mask dimensions must agree.");

	if ((!mxIsDouble(MASK_SCALE) && !mxIsSingle(MASK_SCALE)) || 
	    mxGetNumberOfElements(MASK_SCALE) != 1)
		mexErrMsgTxt("Input mask scale must be a double or single scaler.");
	if ((!mxIsDouble(UNMASK_SCALE) && !mxIsSingle(UNMASK_SCALE)) || 
    	mxGetNumberOfElements(UNMASK_SCALE) != 1)
	    mexErrMsgTxt("Input unmask scale must be a double or single scaler.");
	
	/* Create a matrix for the return argument */
	Y = mxDuplicateArray(X);
	
	nsample = mxGetNumberOfElements(X);
	mask = (bool *) mxGetData(MASK);

    if (mxIsDouble(MASK_SCALE))
    {
       	mask_scale = mxGetScalar(MASK_SCALE);
       	f_mask_scale = mxGetScalar(MASK_SCALE);             // type cast
    }
    else
    {
        mask_scale = *((float *) mxGetData(MASK_SCALE));    // type cast
        f_mask_scale = *((float *) mxGetData(MASK_SCALE));
    }
    
    if (mxIsDouble(UNMASK_SCALE))
    {
       	unmask_scale = mxGetScalar(UNMASK_SCALE);
       	f_unmask_scale = mxGetScalar(UNMASK_SCALE);             // type cast
    }
    else
    {
        unmask_scale = *((float *) mxGetData(UNMASK_SCALE));    // type cast
        f_unmask_scale = *((float *) mxGetData(UNMASK_SCALE));
    }

	if (mxIsDouble(X))
	{
	    y = mxGetPr(Y);
		for (int i = 0; i < nsample; i++)
			y[i] *= (mask[i] ? mask_scale : unmask_scale);
	}
	else
	{
	    f_y = (float *) mxGetData(Y);
		for (int i = 0; i < nsample; i++)
		    f_y[i] *= (mask[i] ? f_mask_scale : f_unmask_scale);
	}
	
	return;
}

