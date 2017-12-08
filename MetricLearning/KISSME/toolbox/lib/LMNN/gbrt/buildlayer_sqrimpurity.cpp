/*This program chooses the best splits, finds the predictions, and computes the loss for the tree.
 */

#include "mex.h"
#include "math.h"
#include <iostream>
#include <string>
#include "matrix.h"

class StaticNode {
public:
    double split;
    double label, loss;
    
    int m_infty, m_s;
    double pxs;
    double l_infty, l_s;

	StaticNode(int m_infty_, double l_infty_) {
		split = 0.0;
		label = 0.0;
		loss = 1.0;
		m_infty = m_infty_;
		m_s = 0;
		pxs = 0.0;
		l_infty = l_infty_;
		l_s = 0.0;
	}
};

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
	// declare variables
    double *Xs,*Xi,*Y,*N;
    double *m_infty, *l_infty, *parents_labels;
	int f;
	double feature_cost;
    
	// check inputs from matlab
	// TODO verify that there are 7 inputs and that each is of the correct size,
	// e.g. size(Xs) == size(Xi) == size(y) == size(n) == [N,1]
	// and size(m_infty)==size(l_infty)==[1,w]
	
	// get data from matlab
    Xs = mxGetPr(prhs[0]);//Sorted Features
    Xi = mxGetPr(prhs[1]);//Indices for Sorted Features
    Y = mxGetPr(prhs[2]);//Labels or Residuals
    N = mxGetPr(prhs[3]);//Nodes
	f = (int) mxGetPr(prhs[4])[0];//Feature Index
	
	// get additional args from matlab
    m_infty = mxGetPr(mxGetCell(prhs[5],0));
    l_infty = mxGetPr(mxGetCell(prhs[5],1));
	parents_labels = mxGetPr(mxGetCell(prhs[5],2));
	feature_cost = mxGetPr(mxGetCell(prhs[5],3))[f-1];
			// added when returning loss for each node
	
	// get dimensions
	int numinstances = mxGetM(prhs[0]);
	int numnodes = mxGetN(mxGetCell(prhs[5],0));
	
	// instantiate parent and child layers of nodes
	StaticNode** parents = new StaticNode*[numnodes];
	StaticNode** children = new StaticNode*[2*numnodes];
	for (int i=0; i<numnodes; i++) {
		parents[i] = new StaticNode((int)m_infty[i], l_infty[i]);
		children[2*i] = new StaticNode(0, 0.0);
		children[2*i]->label = parents_labels[i];
		children[2*i+1] = new StaticNode(0, 0.0);
		children[2*i+1]->label = parents_labels[i];
	}

    // iterate over feature
    for (int j=0; j<numinstances; j++) {
        // get current value
        double v = Xs[j]; // feature value from training set
        int i = (int) Xi[j] - 1; // feature index that corresponds to the unsorted feature
        double l = Y[i]; // target output label for the instance
		int n = (int) N[i] - 1; // node index on the parent layer for the instance
        StaticNode *node = parents[n]; // node on the parent layer for the instance
		
        // if not first instance at node and greater than split point, consider new split at v
        if (node->m_s > 0 && v > node->pxs) {
            double loss_i = -pow(node->l_s,2.0) / (double) node->m_s - pow(node->l_infty - node->l_s,2.0) / (double) (node->m_infty - node->m_s);
            if (node->loss > 0 || loss_i < node->loss) {
                node->loss = loss_i;
                node->split = (node->pxs + v) / 2.0;
                StaticNode* child1 = children[2*n];
                child1->label = node->l_s / (double) node->m_s;
                StaticNode* child2 = children[2*n+1];
                child2->label = (node->l_infty - node->l_s) / (double) (node->m_infty - node->m_s);
            }
        }
		
        // update variables
        node->m_s += 1; // m_s is a counter. The number of data points encountered so far that correspond to that node
        node->l_s += l; // ls is the total residual encountered so far corresponding to that node.
        node->pxs = v; //The previous feature value.
    }
	
    // return splits and loss values for each parent and labels for each child
	/*double *splits = new double[numnodes];
	double *losses = new double[numnodes];
	double *labels = new double[2*numnodes];
	
	for (int i=0; i<numnodes; i++) {
		splits[i] = parents[i]->split;
		losses[i] = parents[i]->loss;
		labels[2*i] = children[2*i]->label;
		labels[2*i+1] = children[2*i+1]->label;
	}
    */
	plhs[0]=mxCreateDoubleMatrix(1,numnodes,mxREAL); // splits
    plhs[1]=mxCreateDoubleMatrix(1,numnodes,mxREAL); // losss
    plhs[2]=mxCreateDoubleMatrix(1,2*numnodes,mxREAL); // labels
    
    double* splits = mxGetPr(plhs[0]);
    double* losses = mxGetPr(plhs[1]);
    double* labels = mxGetPr(plhs[2]);
   
    for (int i=0; i<numnodes; i++) {
		splits[i] = parents[i]->split;
		losses[i] = parents[i]->loss + feature_cost; // add feature cost
		labels[2*i] = children[2*i]->label;
		labels[2*i+1] = children[2*i+1]->label;
	}
	
	// delete parents and children
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
