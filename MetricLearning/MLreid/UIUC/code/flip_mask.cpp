#include <math.h>
#include "mex.h"

/* Input Arguments */
#define	X				prhs[0]
#define	MASK			prhs[1]

/* Output Arguments */
#define	Y				plhs[0]

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	mwSize nsample;
	bool *mask;
	double *y;
	float *f_y;

	/* Check for proper number of arguments */
	if (nrhs != 2)
		mexErrMsgTxt("2 input arguments required.");
	else if (nlhs > 1)
		mexErrMsgTxt("Too many output arguments.");

	/* Check input arguments */
	if (!mxIsDouble(X) && !mxIsSingle(X))
		mexErrMsgTxt("Input data matrix must be double or single.");
	if (!mxIsLogical(MASK))
		mexErrMsgTxt("Input mask matrix must be logical.");
	if (mxGetNumberOfElements(X) != mxGetNumberOfElements(MASK))
		mexErrMsgTxt("Input data and mask dimensions must agree.");

	/* Create a matrix for the return argument */
	Y = mxDuplicateArray(X);

	nsample = mxGetNumberOfElements(X);
	mask = (bool *) mxGetData(MASK);
	if (mxIsDouble(X))
	{
	    y = mxGetPr(Y);
		for (int i = 0; i < nsample; i++)
			if (mask[i]) y[i] = -y[i];
	}
	else
	{
	    f_y = (float *) mxGetData(Y);
		for (int i = 0; i < nsample; i++)
		    if (mask[i]) f_y[i] = -f_y[i];
	}

	return;
}
