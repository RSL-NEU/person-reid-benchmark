/*This program chooses the best splits, finds the predictions, and computes the loss for the tree.
 */

#include "mex.h"
#include "math.h"
#include <iostream>
#include <string>
#include "matrix.h"
#include <omp.h>
#include "float.h"

class StaticNode {
public:
    double split;
    double loss;
    int m_infty, m_s;
    double pxs;
	
    int label_length;
    double* label;
    double* l_infty, *l_s;

	/* Creating a StaticNode instance for the parent where we know m_infty and l_infty. */
	StaticNode(int m_infty_, int label_length_, double *l_infty_, int l_offset) {
		label_length = label_length_;
		m_infty = m_infty_;
        l_infty = new double[label_length];
        for (int j=0; j<label_length; j++) {
        	l_infty[j] = l_infty_[j+l_offset];
        }
		
        label = new double[label_length];
        l_s = new double[label_length];
		clear();
	}
	
	/* Creating an "empty" StaticNode instance for the children. */
	StaticNode(int label_length_) {
		label_length = label_length_;
		m_infty = 0;
        l_infty = new double[label_length];
        for (int j=0; j<label_length; j++) {
        	l_infty[j] = 0.0;
        }
		
        label = new double[label_length];
        l_s = new double[label_length];
		clear();
	}
	

	~StaticNode() {
		delete [] label;
		delete [] l_infty;
		delete [] l_s;  
	}
	
	void clear() {
		split = 0.0;
		loss = DBL_MAX;
		m_s = 0;
		pxs = 0.0;
		for (int j=0; j<label_length; j++) {
			label[j] = 0.0;
			l_s[j] = 0.0;
		}
	}
};

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
	// declare variables
    double *Xs,*Xi,*Y,*N;
    double *m_infty, *l_infty, *parents_labels;
	double *F;
	double *feature_costs;
    
	// check inputs from matlab
	// TODO verify that there are 7 inputs and that each is of the correct size,
	// e.g. size(Xs) == size(Xi) == size(y) == size(n) == [N,1]
	// and size(m_infty)==size(l_infty)==[1,w]
	
	// get data from matlab
    Xs = mxGetPr(prhs[0]);//Sorted Features
    Xi = mxGetPr(prhs[1]);//Indices for Sorted Features
    Y = mxGetPr(prhs[2]);//Labels or Residuals
    N = mxGetPr(prhs[3]);//Nodes
	F = mxGetPr(prhs[4]);//Feature indices
	
	// get additional args from matlab
    m_infty = mxGetPr(mxGetCell(prhs[5],0));
    l_infty = mxGetPr(mxGetCell(prhs[5],1));
	parents_labels = mxGetPr(mxGetCell(prhs[5],2));
	feature_costs = mxGetPr(mxGetCell(prhs[5],3));
			// added when returning loss for each node
	
	
	
	
	//fprintf(stderr,"Here1\n");
	
	// get dimensions
	int numinstances = mxGetM(prhs[0]);
	int numnodes = mxGetN(mxGetCell(prhs[5],0));
	int numfeatures = mxGetM(prhs[4]);
    int label_length = mxGetM(mxGetCell(prhs[5],1));
	
	//fprintf(stderr,"Label length = %d. Size Y = (%d,%d)\n", label_length,mxGetM(prhs[2]),mxGetN(prhs[2]));
	
	
	
	//fprintf(stderr,"Here2\n");
	
	// instantiate outputs
	plhs[0]=mxCreateDoubleMatrix(numfeatures,numnodes,mxREAL); // splits
    plhs[1]=mxCreateDoubleMatrix(numfeatures,numnodes,mxREAL); // losses
    plhs[2]=mxCreateDoubleMatrix(numfeatures,2*numnodes*label_length,mxREAL); // labels
	
    double* splits = mxGetPr(plhs[0]);
    double* losses = mxGetPr(plhs[1]);
    double* labels = mxGetPr(plhs[2]);
	
	
	//fprintf(stderr,"Here3\n");
    // iterate over features
	#pragma omp parallel for default(shared)
	// default(private) shared(Xs,Xi,Y,N,m_infty,l_infty,parents_labels,F,feature_costs)
	for (int f=0; f<numfeatures; f++) {
		
		// instantiate parent and child layers of nodes
		StaticNode** parents = new StaticNode*[numnodes];
		StaticNode** children = new StaticNode*[2*numnodes];
		for (int i=0; i<numnodes; i++) {
			parents[i] = new StaticNode((int)m_infty[i], label_length, l_infty, i*label_length);
			children[2*i] = new StaticNode(label_length);
			for (int j=0; j<label_length; j++) { children[2*i]->label[j] = parents_labels[i*label_length+j]; }
			
			children[2*i+1] = new StaticNode(label_length);
			for (int j=0; j<label_length; j++) { children[2*i+1]->label[j] = parents_labels[i*label_length+j]; }
		}
	
		// consider feature f
		double feature = F[f];
	    for (int j=f*numinstances; j<(f+1)*numinstances; j++) {
			
		
	        // get current value
	        double v = Xs[j]; // feature value from training set
	        int i = (int) Xi[j] - 1; // feature index that corresponds to the unsorted feature
	        int n = (int) N[i] - 1; // node index on the parent layer for the instance
			
			
			StaticNode *node = parents[n]; // node on the parent layer for the instance
		
	        // if not first instance at node and greater than split point, consider new split at v
	        if (node->m_s > 0 && v > node->pxs) {
				//fprintf(stderr,"Here4 - Before computing loss\n");
		
				// compute split impurity
	            double l_s_sqrnorm = 0.0, l_infty_minus_l_s_sqrnorm = 0.0;
				for (int l = 0; l < label_length; l++) {
					l_s_sqrnorm += pow(node->l_s[l],2.0);
					l_infty_minus_l_s_sqrnorm += pow(node->l_infty[l] - node->l_s[l],2.0);
				}
				
				double loss_i = 
					- l_s_sqrnorm / (double) node->m_s 
					- l_infty_minus_l_s_sqrnorm / (double) (node->m_infty - node->m_s) 
					+ feature_costs[(int)feature-1];
				
				
				//if (feature == 465) {
				//	fprintf(stderr, "Weird impurity: l_s_sqrnorm = %f, m_s = %f, l_infty_minus_l_s_sqrnorm = %f, m_infty - m_s = %f, feature_cost = %f \n", l_s_sqrnorm,  (double)node->m_s, l_infty_minus_l_s_sqrnorm, (double) (node->m_infty - node->m_s) , feature_costs[(int)feature-1]);
				//}
				
				//fprintf(stderr,"Here4 - After computing loss\n");
				
				// compare with best and record if better
	            if (loss_i < node->loss) {
					//fprintf(stderr,"Here4 - Before storing\n");
		
	                node->loss = loss_i;
	                node->split = (node->pxs + v) / 2.0;
	                StaticNode* child1 = children[2*n];
					for (int l = 0; l < label_length; l++) {
						child1->label[l] = node->l_s[l] / (double) node->m_s;
					}
						
	                StaticNode* child2 = children[2*n+1];
					for (int l = 0; l < label_length; l++) {
						child2->label[l] = (node->l_infty[l] - node->l_s[l]) / (double) (node->m_infty - node->m_s);
					}
					
					//fprintf(stderr,"Here4 - After storing\n");
		
	            }
				
	        }
	
	        // update variables
	        node->m_s += 1; // m_s is the number of data points encountered so far at that node
			for (int l = 0; l < label_length; l++) { 
				node->l_s[l] += Y[i*label_length + l]; // l_s is the total residual encountered so far at that node.
			} 
			node->pxs = v; // the previous feature value at that node
	    }
	
		//fprintf(stderr,"Here5\n");
		
		// record output values for feature f
	    for (int i=0; i<numnodes; i++) {
			splits[i*numfeatures + f] = parents[i]->split;
			losses[i*numfeatures + f] = parents[i]->loss; // add feature cost
			
			for (int l=0; l < label_length; l++) {
				labels[ (2*i*label_length + l) * numfeatures + f] = children[2*i]->label[l];
				labels[((2*i+1)*label_length + l) * numfeatures + f] = children[2*i+1]->label[l];
			
			}
		}
	
		//fprintf(stderr,"Here6\n");
		
		// delete nodes
		for (int i=0; i<numnodes; i++) {
			delete parents[i];
			parents[i] = NULL;
			delete children[2*i];
			children[2*i] = NULL;
			delete children[2*i+1];
			children[2*i+1] = NULL;
		}
		delete[] parents;
		delete[] children;

	}
}
