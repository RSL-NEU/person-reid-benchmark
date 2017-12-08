/*
 * =============================================================
 * findknnmtree.cpp
  
 * takes two inputs, one (dxn) matrix X and one 
 *  					 (dx1) a test vector
 * 						 
 *                       (1x1) scalar k
 * 
 * and finds the first k nearest neighbors of each column in X
 * (second optional output contains the distances to the k-nn)
 * 
 * 
 * copyright by Kilian Q. Weinberger, 2006
 * kilianw@seas.upenn.edu 
 * =============================================================
 */

/* $Revision: 1.2 $ */

#include "mex.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <stdio.h>



/* If you are using a compiler that equates NaN to zero, you must
 * compile this example using the flag -DNAN_EQUALS_ZERO. For 
 * example:
 *
 *     mex -DAN_EQUALS_ZERO findnz.c  
 *
 * This will correctly define the IsNonZero macro for your
   compiler. */

#if NAN_EQUALS_ZERO
#define IsNonZero(d) ((d) != 0.0 || mxIsNaN(d))
#else
#define IsNonZero(d) ((d) != 0.0)
#endif

#define INPUTX prhs[0]		// data ordered by trees
#define TESTX prhs[1]		// test coordinates
#define NITR prhs[2]			// upper bound in squared distance
#define NITE prhs[3]			// upper bound in squared distance
// tree structure
#define PIVOTSX prhs[4]		// pivots of trees
#define RADIUS prhs[5]		// radi of trees
#define JUMPIND prhs[6]		// interval indices of trees
#define KIDS prhs[7]		// interval indices of trees


#define EXACTINPUTS 8


#include "poppush.cpp"




inline double max(double d1,double d2){
	if(d1<d2) return(d2);
		else return(d1);
}

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




inline double dotp(double *x1,double *x2,int d){
	int i;
	double dist=0;
	
	for(i=0;i<d;i++) {dist+=x1[i]*x2[i];  }
	return(dist);
}


/*
inline double dotp(double *  x1,double *  x2,int d){
	int i=0;
	double dist=0;
	double  d4[4];

	d4[0]=0;d4[1]=0;d4[2]=0;d4[3]=0;	
	for(i=0;i<d-3;i+=4) {
		    d4[0]+=(x1[i+0]*x2[i+0]);
			d4[1]+=(x1[i+1]*x2[i+1]);
			d4[2]+=(x1[i+2]*x2[i+2]);
			d4[3]+=(x1[i+3]*x2[i+3]);
		}

	for(;i<d;i++) {dist+=x1[i]*x2[i];};
	for(int j=0;j<4;j++) dist+=d4[j];
	return(dist);
}

*/


void maxnirecurse(double *MAXNI,double *pKIDS, double *pJI, double *pNITR,int node){
  int kid1=(int) pKIDS[node*2]-1;
  if(kid1==-2){
	double m=0;
	for(int i=(int) pJI[node*2]-1;i<=(int) pJI[node*2+1]-1;i++) 
		if(pNITR[i]>m) m=pNITR[i];
	MAXNI[node]=m;
	
  }	 else {
		int kid2=(int) pKIDS[node*2+1]-1;	 
		maxnirecurse(MAXNI,pKIDS, pJI, pNITR,kid1);
		maxnirecurse(MAXNI,pKIDS, pJI, pNITR,kid2);
		MAXNI[node]=max(MAXNI[kid1],MAXNI[kid2]);				
  }	
}



void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  /* Declare variables. */ 
  int DIM,N,Ni,MAXTREE,node,MAXIMPS;
  double *pX, *pTESTX, *pPIVX,*pRADI;
  double *pJI,*pKIDS, *pNITR, *pNITE;
  int oc=0;



  /* Check for proper number of input and output arguments. */    
  if (nrhs != EXACTINPUTS) {
	mexErrMsgTxt("Not the right number of inputs.\nPlease call findknnmtree(x,testx,k,tree.pivots,tree.radius,tree.jumpindex,tree.kids);");
  } 
  if (nlhs > 3) {
    mexErrMsgTxt("Too many output arguments.");
  }

  /* Get the data. */
  pX  		= (double *)mxGetPr(INPUTX);
  pTESTX  	= (double *)mxGetPr(TESTX);
  pPIVX  	= (double *)mxGetPr(PIVOTSX);
  pRADI  	= (double *)mxGetPr(RADIUS);
  pJI  		= (double *)mxGetPr(JUMPIND);
  pKIDS  	= (double *)mxGetData(KIDS);
  pNITR	  	= (double *)mxGetData(NITR);
  pNITE	  	= (double *)mxGetData(NITE);



//  printf("K=%i\n",k);

  /* Get the number of elements in the input argument. */
  DIM 		= mxGetM(INPUTX);
  N 		= mxGetN(TESTX);
  Ni 		= mxGetN(INPUTX);
  MAXTREE 	= mxGetN(PIVOTSX);



  if (mxGetN(NITE)!=mxGetN(TESTX)) {
    mexErrMsgTxt("There must be one cutoff value for each testpoint.");
  }
  if (mxGetN(NITR)!=mxGetN(INPUTX)) {
    mexErrMsgTxt("There must be one cutoff value for each trainingpoint.");
  }

 // define output matrix
 MAXIMPS=10*N*2;
 plhs[0]=mxCreateDoubleMatrix(2,MAXIMPS,mxREAL);
 double * po=mxGetPr(plhs[0]);


  // here should be some checkups if all the variables have the right size//
 poppush_t stack=poppush();
 stack.init(MAXTREE);

 
 // compute MAXNI 
 double* MAXNI = new double[MAXTREE];
 
 maxnirecurse(MAXNI,pKIDS, pJI, pNITR,0);

 double rejected=0,taken=0;
	
 for(int input=0;input<N;input++)	{	
 // initialize abstract datatype of stack
 int indim=input*DIM;
 double sni=pNITE[input];
 double sqni=sqrt(pNITE[input]);

 
// compute first radius
 double md0=max(sqrt(distance(&pPIVX[0],&pTESTX[indim],DIM))-pRADI[0],0);
 if(md0<sqni || (md0*md0)<MAXNI[0]){
  stack.push(0,0); // push first tree onto stack
  while(1){
	double mindist;
	int fb;
//	do{
  	  fb=stack.pop(&node,&mindist);	 
//     } while(fb>=0 && mindist>sqni && mindist>MAXNI[node]);
	if(fb==-1)	break;
	
	int kid1=(int) pKIDS[node*2]-1;
	 
	
	if(kid1<0) { // leaf
		for(int i=(int) pJI[node*2]-1;i<=(int) pJI[node*2+1]-1;i++){
			double cutoff=max(sni,pNITR[i]);
			double dist=distance(&pX[DIM*i],&pTESTX[indim],DIM,cutoff);		
  			if(dist<cutoff) {
						po[oc]=input+1;
						po[oc+1]=i+1;
						oc+=2;
                        if(oc>=MAXIMPS-1){      
							printf(".");
                            double *temp=po;
							MAXIMPS=(int) ceil((((double) MAXIMPS)/2)*(N+1)/(input+1)*1.1)*2;
							mxArray *mtemp=plhs[0];
							plhs[0]=mxCreateDoubleMatrix(2,MAXIMPS,mxREAL);                             
                            po=mxGetPr(plhs[0]);
                            memcpy(po,temp,oc*sizeof(double));                            
							mxDestroyArray(mtemp);
                        }
					}
		}
	}
	else
	{ // no leaf
	 int kid2=(int) pKIDS[node*2+1]-1;	
     
 	 double md1=max(sqrt(distance(&pPIVX[DIM*kid1],&pTESTX[indim],DIM))-pRADI[kid1],0);
	 if(md1<sqni || md1*md1<MAXNI[kid1]) stack.push(kid1,md1);
     
          
	 double md2=max(sqrt(distance(&pPIVX[DIM*kid2],&pTESTX[indim],DIM))-pRADI[kid2],0);	
	 // push children onto stack
 	 if(md2<sqni || md2*md2<MAXNI[kid2]) stack.push(kid2,md2);
    }
   }
  }
 }

 stack.cleanup();
 mxSetN(plhs[0],(int) ceil ((double)oc/2));  
}

