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
#include <omp.h>
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
#define INPUTK prhs[2]			// number of nearest neighbors
// tree structure
#define PIVOTSX prhs[3]		// pivots of trees
#define RADIUS prhs[4]		// radi of trees
#define JUMPIND prhs[5]		// interval indices of trees
#define KIDS prhs[6]		// interval indices of trees


#define EXACTINPUTS 7

#include "heaptree.cpp"
#include "poppush.cpp"




inline double max(double d1,double d2){
	if(d1<d2) return(d2);
		else return(d1);
}

inline double distance(const double *  x1,const double *  x2,int d){
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

inline double distance(const double *     x1,const double *   x2,int d, double cutoff){
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




inline double dotp(const double *x1,const double *x2,int d){
	int i;
	double dist=0;
	
	for(i=0;i<d;i++) {dist+=x1[i]*x2[i];  }
	return(dist);
}

struct tree
/* internal structure for stack */
{
	double *pPIVX;
	double *pRADIUS;
	double *pJI;
	double *pKIDS;
	double *pINDEX;
	int MAXTREE;
};


void findknnmtree(int input,double *outknn, double *outdist, const double *pX,const double *pTESTX,const struct tree *t1,const int Ni,const int N,const int DIM, const int k){	
	int node;
 // here should be some checkups if all the variables have the right size//

 double double_max=2147483647;
 double double_max2=double_max*2;
 while(double_max<double_max2){     
     double_max=double_max2;
     double_max2=double_max*2;     
 }
 
	
 // for(int input=0;input<N;input++)	
{	
 // initialize abstract datatype of stack
 int indim=input*DIM;
 poppush_t stack=poppush();
 stack.init(t1->MAXTREE);
 stack.push(0,0); // push first tree onto stack

 // Init heaptree
 heaptree_t heap=heaptree();
 heap.initheaptree(1,k);

  while(1){
	double mindist;
	int fb;
	do{
	 fb=stack.pop(&node,&mindist);	 
     } while(heap.heapsize[0]==k && fb>=0 && mindist>heap.heapnodes[0][0]);
	if(fb==-1)	break;
	
	int kid1=(int) t1->pKIDS[node*2]-1;
	 
	
	if(kid1<0) { // leaf
		double thresh=heap.heapnodes[0][0]*heap.heapnodes[0][0]; // avoid some sqrt() operations
		if(heap.heapsize[0]<k) thresh=double_max;
		for(int i=(int) t1->pJI[node*2]-1;i<=(int) t1->pJI[node*2+1]-1;i++){
			double dist=distance(&pX[DIM*i],&pTESTX[indim],DIM,thresh);	
  			if(dist<thresh) {
					heap.heapupdate(0,sqrt(dist),i+1);
					thresh=heap.heapnodes[0][0]*heap.heapnodes[0][0];
					if(heap.heapsize[0]<k) thresh=double_max;
					}
		}
	}
	else
	{ // no leaf
	 int kid2=(int) t1->pKIDS[node*2+1]-1;		
 	 double d1=sqrt(distance(&t1->pPIVX[DIM*kid1],&pTESTX[indim],DIM));
	 double d2=sqrt(distance(&t1->pPIVX[DIM*kid2],&pTESTX[indim],DIM));
	
	 if(d1<d2) {	// if kid1 is the closer child
	  	 stack.push(kid2,max(d2-t1->pRADIUS[kid2],0));
	 	 stack.push(kid1,max(d1-t1->pRADIUS[kid1],0));
  	 } else{		// if kid2 is the closer child
		 stack.push(kid1,max(d1-t1->pRADIUS[kid1],0));
  	 	 stack.push(kid2,max(d2-t1->pRADIUS[kid2],0));	
	}
  }
  }
 stack.cleanup();
 int i=(input+1)*k-1;
  for(int m=k-1;m>=0;m--){
   outknn[i]=heap.heapdata[0][0];
   outdist[i]=heap.heapnodes[0][0];
   heap.heappoproot(0);
   i--;
  }
 heap.cleanup(1);	
 }
}



void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  /* Declare variables. */ 
  int k,DIM,N,Ni;
  double *pX, *pTESTX;
  tree outtree;

  /* Check for proper number of input and output arguments. */    
  if (nrhs != EXACTINPUTS) {
	mexErrMsgTxt("Not the right number of inputs.\nPlease call findknnmtree(x,testx,k,tree.pivots,tree.radius,tree.jumpindex,tree.kids);");
  } 
  if (nlhs > 3) {
    mexErrMsgTxt("Too many output arguments.");
  }

  /* Get the data. */
  double *pk= (double *)mxGetPr(INPUTK); k=(int) pk[0];
  pX  		= (double *)mxGetPr(INPUTX);
  pTESTX  	= (double *)mxGetPr(TESTX);
  outtree.pPIVX  	= (double *)mxGetPr(PIVOTSX);
  outtree.pRADIUS  	= (double *)mxGetPr(RADIUS);
  outtree.pJI  		= (double *)mxGetPr(JUMPIND);
  outtree.pKIDS  	= (double *)mxGetData(KIDS);


//  printf("K=%i\n",k);

  /* Get the number of elements in the input argument. */
  DIM 		= mxGetM(INPUTX);
  N 		= mxGetN(TESTX);
  Ni 		= mxGetN(INPUTX);
  outtree.MAXTREE 	= mxGetN(PIVOTSX);

  // define output pointers
  plhs[0]=mxCreateDoubleMatrix(k,N,mxREAL);
  plhs[1]=mxCreateDoubleMatrix(k,N,mxREAL);
  double * po=mxGetPr(plhs[0]);
  double *po2=mxGetPr(plhs[1]);

  // finally build tree
#pragma omp parallel shared(outtree) //private(kstart,kend,i,j,m,insert)
  {
#pragma omp for
  for(int input=0;input<N;input++)	
    findknnmtree(input,po, po2, pX, pTESTX, &outtree, Ni,N, DIM, k);
  }
}

