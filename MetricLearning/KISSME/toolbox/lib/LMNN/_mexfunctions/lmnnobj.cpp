/*
 * =============================================================
 * cdist.c 
  
 * input: x,a,b
 * x: DxN matrix
 * a: 1xN vector of indices
 * b: 1xN vector of indices
 * 
 * output: sum((x(:,a)-x(:,b)).^2)
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

#define X_IN prhs[0]
#define T_IN prhs[1]
#define I_IN prhs[2]

double square(double x) { return(x*x);}



double dotp(int m, double *v1,double *v2){
  int j;
  double dp=0;

   for(j=0;j<m;j++){ 
     dp=dp+v1[j]*v2[j];
   }
   return (dp);
};


inline double distance(double *  x1,double *  x2,int d){
	int i=0;
	double  d4[4];

	d4[0]=0;d4[1]=0;d4[2]=0;d4[3]=0;	
	for(i=0;i<d-3;i+=4) {
			d4[0]+=(x1[i+0]-x2[i+0])*(x1[i+0]-x2[i+0]);
			d4[1]+=(x1[i+1]-x2[i+1])*(x1[i+1]-x2[i+1]);
			d4[2]+=(x1[i+2]-x2[i+2])*(x1[i+2]-x2[i+2]);
			d4[3]+=(x1[i+3]-x2[i+3])*(x1[i+3]-x2[i+3]);
		}
	for(;i<d;i++) {double r=x1[i]-x2[i]; d4[2]+=r*r; }
	d4[0]+=d4[1];
	d4[2]+=d4[3];
	return(d4[0]+d4[2]);
}

inline double distance(double *     x1,double *   x2,int d, double cutoff){
	double dist=0;
	
	int i=0;
	for(;i<d-3;i+=4) {
		double  r1=(x1[i]-x2[i])*(x1[i]-x2[i]); 
		double  r2=(x1[i+1]-x2[i+1])*(x1[i+1]-x2[i+1]); 
		double  r3=(x1[i+2]-x2[i+2])*(x1[i+2]-x2[i+2]); 
		double  r4=(x1[i+3]-x2[i+3])*(x1[i+3]-x2[i+3]); 		
		dist+=r1+r2+r3+r4; 
		if(dist>cutoff) return(dist);
	}
	for(;i<d;i++) {
		double r1=x1[i]-x2[i]; 
		dist+=r1*r1;
	}
	return(dist);
}


double computeloss(double *X, short int *T, short int *I, int d, int kt, int ki,int i,double *grad){
    
    double *dt = new double[kt];
    double lossT=0.0;
    double lossI=0.0;
    double ma=0.0;
        
    // compute distances to target neighbors
     for(int k=0; k<kt; k++) {
         dt[k]=distance(&X[i*d],&X[(T[k]-1)*d],d)+1.0;
         lossT+=dt[k]-1.0;
         if (dt[k]>ma) ma=dt[k];
         
         // update gradient
         for(int j=0;j<d;j++){
              grad[i*d+j]+=X[i*d+j]-X[(T[k]-1)*d+j];
              grad[(T[k]-1)*d+j]-=X[i*d+j]-X[(T[k]-1)*d+j];
         }
     }
    delete[] dt;
    // compute distances to impostors
    for(int k=0; k<ki; k++) {
         double dis=distance(&X[i*d],&X[(I[k]-1)*d],d,ma);
         for(int t=0; t<kt; t++) if (dt[t]>dis) {
           lossI+=dt[t]-dis;        
                             
           // update gradient
           for(int j=0;j<d;j++){
              grad[i*d+j]       -=X[(T[t]-1)*d+j]   -X[(I[k]-1)*d+j];
              grad[(T[t]-1)*d+j]-=X[i*d+j]          -X[(T[t]-1)*d+j];
              grad[(I[k]-1)*d+j]+=X[i*d+j]          -X[(I[k]-1)*d+j];
           }
//            grad(:,i) = grad(:,i)-mu*(pred(:,targets_i)-pred(:,impos_i(k)));
//            grad(:,targets_i) = grad(:,targets_i)-mu*(pred(:,i)-pred(:,targets_i));
//            grad(:,impos_i(k)) = grad(:,impos_i(k))+mu*(pred(:,i)-pred(:,impos_i(k)));

         }
    }

    return(0.5*(lossT+lossI));
    
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{

  /* Check for proper number of input and output arguments. */    
  if (nrhs != 3) {
    mexErrMsgTxt("Exactly three input arguments required.");
  } 

  /* Check data type of input argument. */
  if (!(mxIsDouble(X_IN))) {
   mexErrMsgTxt("Input array must be of type double.");
  }
      
   if(mxGetN(I_IN)!=mxGetN(T_IN) || mxGetN(T_IN)!=mxGetN(X_IN))
    mexErrMsgTxt("X, Target neighbors and Impostors must have same number of columns\n");


  double *X=mxGetPr(X_IN); // X data
  short int *T=(short int*) mxGetData(T_IN); // Target neighbors
  short int *I=(short int*) mxGetData(I_IN); // Impostors
  
  /* Get the data. */
  int n = mxGetN(X_IN); // number of inputs
  int d = mxGetM(X_IN); // dimension of inputs
  int kt = mxGetM(T_IN); // number of target neighbors
  int ki = mxGetM(I_IN); // number of impostors
 
  /* Create output matrix for loss */
  plhs[0]=mxCreateDoubleMatrix(1,n,mxREAL);
  double *loss=mxGetPr(plhs[0]);

  
  /* Create output matrix for gradient */
  plhs[1]=mxCreateDoubleMatrix(d,n,mxREAL);
  double *grad=mxGetPr(plhs[1]);
  

  for (int i=0;i<n;i++)
      loss[i]=computeloss(X,&T[i*kt],&I[i*ki],d,kt,ki,i,grad);
  
}


