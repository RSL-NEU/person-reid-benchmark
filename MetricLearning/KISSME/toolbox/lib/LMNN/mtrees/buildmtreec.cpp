/*
 * =============================================================
 * buildmtree.cpp
  
 * takes two inputs, one (dxn) matrix X and one 
 *  					 (1x1) scalar (maximum number of points in leaf)
 * 						 
 * 			output: 	 (nx1) new indices of points (random permutation of inputs)
 *                       (dxl) locations of pivots
 *						 (1xl) radius of trees
 * 						 (2xl) jumpindex of trees (tree i goes from ji(1,i):ji(2,i))
 * 						 (2x1) kids	 of trees (tree i has kids kids(1,i) and kids(2,i))
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
#include <stack>
#include <vector>

using namespace std;

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

#define MINRADIUS 0.0001

// define output
#define INPUTX prhs[0]		// data ordered by trees
#define INPUTK prhs[1]		// number of nearest neighbors

#define EXACTINPUTS 2


// define output
#define INDEX plhs[0]		// test coordinates
// tree structure
#define PIVOTSX plhs[1]		// pivots of trees
#define RADIUS plhs[2]		// radi of trees
#define JUMPIND plhs[3]		// interval indices of trees
#define KIDS plhs[4]		// interval indices of trees

#define EXACTOUTPUTS 5


#include "heaptree.cpp"
//#import "poppushbt.cpp"


struct tree
/* internal structure for stack */
{
	double *pPIVX;
	double *pRADIUS;
	double *pJI;
	double *pKIDS;
	double *pINDEX;
	unsigned int MAXTREE;
};




struct cell
/* internal structure for stack */
{
    int number;
    int ij[2];
};



inline double max(double d1,double d2)
/* max(d1,d2) */
{
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



inline void vecminus(double *m,double *x1,double *x2,int d)
/* vector operation: m=x1-x2 */
{
	for(int i=0;i<d;i++) m[i]=x1[i]-x2[i]; 
}

inline void vecplus(double *m,double *x1,double *x2,int d)
/* vector operation: m=x1+x2 */
{
	for(int i=0;i<d;i++) m[i]=x1[i]+x2[i]; 
}

inline double dotp(double *  x1,double *  x2,int d)
/* compute the dot product between vector x1 and x2 */
{
	int i=0;
	double  d4[4];
	double dist=0;

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


void mean(double *x,double *m,int dim,int i1,int i2,int *index){
/* 
  computes the mean of a matrix x with dimensions dxn
  m  must be of dimensions dx1
*/		
	for(int j=0;j<dim;j++) m[j]=0;  		// fill target vectors with zeros

	for(int i=i1;i<=i2;i++)	
		vecplus(m,m,&x[index[i]*dim],dim);	// sum over all vectors

	for(int j=0;j<dim;j++) m[j]/=(i2-i1+1);			// average
}


inline double compradius(double *x,double *piv,int dim,int i1,int i2,int *index){
/* finds the maximum L2 distance between vector piv and all rows in matrix x */
	double dist=0;
	
	for(int i=i1;i<=i2;i++){
		double ds=distance(&x[index[i]*dim],piv,dim);
		if(ds>dist) dist=ds;
	}
	return(sqrt(dist));
}


inline int compfurthest(double *x,double *piv,int dim,int i1,int i2,int *index){
/* finds the maximum L2 distance between vector piv and all rows in matrix x */
	int furthest=0;
	double dist=0;
	
	for(int i=i1;i<=i2;i++){		
		double ds=distance(&x[index[i]*dim],piv,dim);
		if(ds>dist) {dist=ds;furthest=i;}
	}
	return(furthest);
}



void populatetree(double *pX,struct tree *t1,int N,int DIM,int mi)
{
 /* Initialize some variables */
 int *index	= new int[N];
 for(int i=0;i<N;i++) index[i]=i;

 t1->MAXTREE=0;
 stack <cell,vector<cell> > s;  	// stack

 // Initialize first cell
 cell c;
 c.number=t1->MAXTREE;
 c.ij[0]=0;c.ij[1]=N-1;
 s.push(c);
	while(s.size()!=0){				
		c=s.top(); s.pop();										// pop first element from stack
		int i1=c.ij[0], i2=c.ij[1];								// set first and last index
		int ni=i2-i1+1;											// compute length of interval
		double* piv = new double[DIM];										// get memory for pivot
		mean(pX,piv,DIM,i1,i2,index);					// compute pivot point

		double radius=compradius(pX,piv,DIM,i1,i2,index);	// compute radius of ball
		
		if(ni<mi || radius<MINRADIUS){
			// if tree has fewer than mi data points, make it a leaf
			int two=2*c.number, one=c.number;
			t1->pKIDS[two]=-1; t1->pKIDS[two+1]=-1;			// indicate leaf node (through -1 kids)
			t1->pJI[two]=i1+1;t1->pJI[two+1]=i2+1;
			t1->pRADIUS[one]=radius;
			memcpy(&t1->pPIVX[c.number*DIM],piv,DIM*sizeof(double));
		} else
		{
			// compute statistics about pivot points
			int two=2*c.number, one=c.number;
			t1->pKIDS[two]=t1->MAXTREE+2; t1->pKIDS[two+1]=t1->MAXTREE+3;		// indicate leaf node (through -1 kids)
			t1->pJI[two]=i1+1;t1->pJI[two+1]=i2+1;
			t1->pRADIUS[one]=radius;
			memcpy(&t1->pPIVX[c.number*DIM],piv,DIM*sizeof(double));
			
			// pick two points that are far away from each other 
			int r=i1+(int) floor(rand()/((double)RAND_MAX + 1)*ni);			
			int p1=compfurthest(pX,&pX[index[r ]*DIM],DIM,i1,i2,index);
			int p2=compfurthest(pX,&pX[index[p1]*DIM],DIM,i1,i2,index);

			// compute dir=(x1-x2)  and project each point onto this direction
			double* dir = new double[DIM];
			vecminus(dir,&pX[index[p1]*DIM],&pX[index[p2]*DIM],DIM);

            // compute the inner products (ips)
            double* ips = new double[ni];
            double meanip=0;
			for(int i=i1;i<=i2;i++){
				ips[i-i1]=dotp(&pX[index[i]*DIM],dir,DIM);
                meanip+=ips[i-i1];
			}
            meanip/=ni;
			// decide if the projected point is closer to pivot 1 or pivot 2            
			int* ind1 = new int[ni];
            int* ind2 = new int[ni];
            int c1=0,c2=0;
            for(int i=i1;i<=i2;i++){
				if(ips[i-i1]>meanip){ // belongs to pivot 1
					ind1[c1]=index[i];
					c1++;
				} else {  // belongs to pivot 2
					ind2[c2]=index[i];
					c2++;										
				}				                
            }
			// sort the index 
			for(int i=0;i<c1;i++) index[i1+i   ]=ind1[i];
			for(int i=0;i<c2;i++) index[i1+i+c1]=ind2[i];
			
			// Prevent potential infinite loop
			if(c1==0 || c2==0) mexErrMsgTxt("A subtree with 0 elements was created. This should never happen!");		

			// push subtree 1 onto the stack
			cell cell1;
			cell1.number=t1->MAXTREE+1;
			cell1.ij[0]=i1;
			cell1.ij[1]=i1+c1-1;
			s.push(cell1);

			// push subtree 2 onto the stack
			cell cell2;
			cell2.number=t1->MAXTREE+2;
			cell2.ij[0]=i1+c1;
			cell2.ij[1]=i2;
			s.push(cell2);

			t1->MAXTREE+=2;
		}

	}

	for(int i=0;i<N;i++)
		t1->pINDEX[i]=(int) index[i]+1;
}





void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  /* Declare variables. */ 
  int mi,DIM,N;
  double *pX;

  struct tree outtree;
 

  /* Check for proper number of input and output arguments. */    
  if (nrhs != EXACTINPUTS) {
	mexErrMsgTxt("Not the right number of inputs.\nPlease call findknnmtree(x,testx,k,tree.pivots,tree.radius,tree.jumpindex,tree.kids);");
  } 
  if (nlhs != EXACTOUTPUTS) {
    mexErrMsgTxt("Needs 5 output arguments!");
  }

  /* Get the data. */
  double *pk= (double *)mxGetPr(INPUTK); mi=(int) pk[0];
  pX  		= (double *)mxGetPr(INPUTX);

  /* Get the number of elements in the input argument. */
  DIM 		= mxGetM(INPUTX);
  N 		= mxGetN(INPUTX);

 /* Create output pointers */
 INDEX		= mxCreateDoubleMatrix(1,N,mxREAL);
 PIVOTSX	= mxCreateDoubleMatrix(DIM,N,mxREAL);
 RADIUS		= mxCreateDoubleMatrix(1,N,mxREAL);
 JUMPIND	= mxCreateDoubleMatrix(2,N,mxREAL);
 KIDS		= mxCreateDoubleMatrix(2,N,mxREAL);

 outtree.pPIVX  	= (double *)mxGetPr(PIVOTSX);
 outtree.pRADIUS  	= (double *)mxGetPr(RADIUS);
 outtree.pJI  		= (double *)mxGetPr(JUMPIND);
 outtree.pKIDS  	= (double *)mxGetPr(KIDS);
 outtree.pINDEX		= (double *)mxGetPr(INDEX);

 populatetree(pX,&outtree,N,DIM,mi);

 // cut off zeros
  mxSetN(RADIUS,outtree.MAXTREE+1);  
  mxSetN(JUMPIND,outtree.MAXTREE+1);  
  mxSetN(PIVOTSX,outtree.MAXTREE+1);  
  mxSetN(KIDS,outtree.MAXTREE+1);  
}









