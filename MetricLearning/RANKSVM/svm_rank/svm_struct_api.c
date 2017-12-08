/***********************************************************************/
/*                                                                     */
/*   svm_struct_api.c (instantiated for SVM-rank)                      */
/*                                                                     */
/*   Definition of API for attaching implementing SVM learning of      */
/*   structures (e.g. parsing, multi-label classification, HMM)        */ 
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 15.03.09                                                    */
/*                                                                     */
/*   Copyright (c) 2005  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "svm_struct_api.h"
#include "svm_light/svm_common.h"
#include "svm_struct/svm_struct_common.h"
#include "svm_struct/svm_struct_learn.h"

#define MAX(x,y)      ((x) < (y) ? (y) : (x))
#define MIN(x,y)      ((x) > (y) ? (y) : (x))
#define SIGN(x)       ((x) > (0) ? (1) : (((x) < (0) ? (-1) : (0))))

int compareup(const void *a, const void *b) 
{
  double va,vb;
  va=((STRUCT_ID_SCORE *)a)->score;
  vb=((STRUCT_ID_SCORE *)b)->score;
  if(va == vb) {
    va=((STRUCT_ID_SCORE *)a)->tiebreak;
    vb=((STRUCT_ID_SCORE *)b)->tiebreak;
  }
  return((va > vb) - (va < vb));
}

int comparedown(const void *a, const void *b) 
{
  return(-compareup(a,b));
}

double swappedpairs(LABEL y, LABEL ybar);
double fracswappedpairs(LABEL y, LABEL ybar);


void        svm_struct_learn_api_init(int argc, char* argv[])
{
  /* Called in learning part before anything else is done to allow
     any initializations that might be necessary. */
}

void        svm_struct_learn_api_exit()
{
  /* Called in learning part at the very end to allow any clean-up
     that might be necessary. */
}

void        svm_struct_classify_api_init(int argc, char* argv[])
{
  /* Called in prediction part before anything else is done to allow
     any initializations that might be necessary. */
}

void        svm_struct_classify_api_exit()
{
  /* Called in prediction part at the very end to allow any clean-up
     that might be necessary. */
}

SAMPLE      read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm)
{
  /* Reads struct examples and returns them in sample. The number of
     examples must be written into sample.n */
  SAMPLE   sample;  /* sample */
  EXAMPLE  *examples;
  PATTERN  *x;
  LABEL    *y;
  long     n,qid;       /* number of instances */
  long     totwords, totpairs, sumtotpairs, i, j, k;
  double   *labels;
  DOC      **instances;

  /* Using the read_documents function from SVM-light */
  if(0) {
   /* we have only one big example */
    examples=(EXAMPLE *)my_malloc(sizeof(EXAMPLE));
    read_documents(file,&examples[0].x.doc,&examples[0].y._class,&totwords,&n);
    examples[0].x.totdoc=n;
    examples[0].y.totdoc=n;
    sample.n=1;
    sample.examples=examples;
  }
  else{
    read_documents(file,&instances,&labels,&totwords,&n);
    qid=-1;
    sample.n=0;
    examples=NULL;
    x=NULL;
    y=NULL;
    for(i=0;i<n;i++) {
      if(instances[i]->queryid < 0) {
		printf("ERROR: Query ID's in data file have to be positive!\n");
		exit(1);
      }
      if(instances[i]->queryid != qid) {
		if(instances[i]->queryid < qid) {
		  printf("ERROR: Query ID's in data file have to be in increasing order (line %ld)!\n",i);
		  exit(1);
		}
		qid=instances[i]->queryid;
		sample.n++;
		examples=(EXAMPLE *)realloc(examples,sizeof(EXAMPLE)*sample.n);
		x=&examples[sample.n-1].x;
		y=&examples[sample.n-1].y;
		x->doc=NULL;
		x->totdoc=0;
		y->factor=NULL;
		y->_class=NULL;
		y->loss=0;
        y->totdoc=0;
      }
      x->totdoc++;
      y->totdoc++;
      x->doc=(DOC **)realloc(x->doc,sizeof(DOC *)*x->totdoc);
      x->doc[x->totdoc-1]=instances[i];
      y->_class=(double *)realloc(y->_class,sizeof(double)*y->totdoc);
      y->_class[y->totdoc-1]=labels[i];
    }
    sample.examples=examples;
    free(instances);
    free(labels);
  }

  /* Remove all features with numbers larger than num_features, if
     num_features is set to a positive value. This is important for
     svm_struct_classify. */
  if(sparm->num_features > 0) 
    for(k=0;k<sample.n;k++) 
      for(i=0;i<sample.examples[k].x.totdoc;i++)
	for(j=0;sample.examples[k].x.doc[i]->fvec->words[j].wnum;j++) 
	  if(sample.examples[k].x.doc[i]->fvec->words[j].wnum>sparm->num_features) {
	    sample.examples[k].x.doc[i]->fvec->words[j].wnum=0;
	    sample.examples[k].x.doc[i]->fvec->words[j].weight=0;
	  }

  /* add label factors for easy computation of feature vectors */
  sumtotpairs=0;
  for(k=0;k<sample.n;k++) {
    x=&sample.examples[k].x;
    y=&sample.examples[k].y;
    y->factor=(double *)my_malloc(sizeof(double)*y->totdoc);
    for(i=0;i<y->totdoc;i++) 
      y->factor[i]=0;
    totpairs=0;
    for(i=0;i<y->totdoc;i++) {
      for(j=0;j<y->totdoc;j++) {
	if(y->_class[i] > y->_class[j]) {
	  totpairs++;
	  y->factor[i]+=0.5;
	  y->factor[j]-=0.5;
	}
      }
    }
    sumtotpairs+=totpairs;
    x->scaling=1.0/(double)totpairs; /* for FRACSWAPPEDPAIRS */
  }

  /* modify scaling factor */
  if(sparm->loss_function == SWAPPEDPAIRS) {
    sparm->epsilon*=(double)sumtotpairs/(double)sample.n; /* adjusting eps */
    for(k=0;k<sample.n;k++) {
      x=&sample.examples[k].x;
      x->scaling=1.0;
    }
  }
  else if(sparm->loss_function == FRACSWAPPEDPAIRS) {
    /* already set to correct value above */
  }
  else {
    printf("Unknown loss function type: %d\n",sparm->loss_function);
    exit(1);
  }
  for(k=0;k<sample.n;k++) {
    x=&sample.examples[k].x;
    y=&sample.examples[k].y;
    for(i=0;i<y->totdoc;i++) 
      y->factor[i]*=x->scaling;
  }

  return(sample);
}

void        init_struct_model(SAMPLE sample, STRUCTMODEL *sm, 
			      STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, 
			      KERNEL_PARM *kparm)
{
  /* Initialize structmodel sm. The weight vector w does not need to be
     initialized, but you need to provide the maximum size of the
     feature space in sizePsi. This is the maximum number of different
     weights that can be learned. Later, the weight vector w will
     contain the learned weights for the model. */
  long   i,k,totwords=0,totdoc=0;
  WORD   *w;

  totwords=0;  /* find highest feature number */
  totdoc=0;
  for(k=0;k<sample.n;k++)
    for(i=0;i<sample.examples[k].x.totdoc;i++) {
      totdoc++;
      for(w=sample.examples[k].x.doc[i]->fvec->words;w->wnum;w++) 
	if(totwords < w->wnum) 
	  totwords=w->wnum;
    }
  sparm->num_features=totwords;
  if(struct_verbosity>=0) {
    printf("Training set properties: %d features, %d rankings, %ld examples\n",
	   sparm->num_features,sample.n,totdoc);
    if(sparm->loss_function == SWAPPEDPAIRS) 
      printf("NOTE: Adjusted stopping criterion relative to maximum loss: eps=%lf\n",sparm->epsilon);
  }
  sm->sizePsi=sparm->num_features;
  if(struct_verbosity>=2)
    printf("Size of Phi: %ld\n",sm->sizePsi);
}

CONSTSET    init_struct_constraints(SAMPLE sample, STRUCTMODEL *sm, 
				    STRUCT_LEARN_PARM *sparm)
{
  /* Initializes the optimization problem. Typically, you do not need
     to change this function, since you want to start with an empty
     set of constraints. However, if for example you have constraints
     that certain weights need to be positive, you might put that in
     here. The constraints are represented as lhs[i]*w >= rhs[i]. lhs
     is an array of feature vectors, rhs is an array of doubles. m is
     the number of constraints. The function returns the initial
     set of constraints. */
  CONSTSET c;

  if(1) { /* normal case: start with empty set of constraints */
    c.lhs=NULL;
    c.rhs=NULL;
    c.m=0;
  }
  return(c);
}

LABEL       classify_struct_example(PATTERN x, STRUCTMODEL *sm, 
				    STRUCT_LEARN_PARM *sparm)
{
  /* Finds the label yhat for pattern x that scores the highest
     according to the linear evaluation function in sm, especially the
     weights sm.w. The returned label is taken as the prediction of sm
     for the pattern x. The weights correspond to the features defined
     by psi() and range from index 1 to index sm->sizePsi. If the
     function cannot find a label, it shall return an empty label as
     recognized by the function empty_label(y). */
  LABEL y;
  int i;

  y.totdoc=x.totdoc;
  y._class=(double *)my_malloc(sizeof(double)*y.totdoc);
  y.factor=NULL;
  y.loss=-1;
  /* simply classify by sign of inner product between example vector
     and weight vector */
  for(i=0;i<x.totdoc;i++) {
    y._class[i]=classify_example(sm->svm_model,x.doc[i]);
  }
  return(y);
}

LABEL       find_most_violated_constraint_slackrescaling(PATTERN x, LABEL y, 
						     STRUCTMODEL *sm, 
						     STRUCT_LEARN_PARM *sparm)
{
  /* Finds the label ybar for pattern x that that is responsible for
     the most violated constraint for the slack rescaling
     formulation. It has to take into account the scoring function in
     sm, especially the weights sm.w, as well as the loss
     function. The weights in sm.w correspond to the features defined
     by psi() and range from index 1 to index sm->sizePsi. Most simple
     is the case of the zero/one loss function. For the zero/one loss,
     this function should return the highest scoring label ybar, if
     ybar is unequal y; if it is equal to the correct label y, then
     the function shall return the second highest scoring label. If
     the function cannot find a label, it shall return an empty label
     as recognized by the function empty_label(y). */
  LABEL ybar;
  printf("ERROR: Slack-rescaling is not implemented for this loss function!\n");
  exit(1);
  return(ybar);
}

LABEL       find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, 
						     STRUCTMODEL *sm, 
						     STRUCT_LEARN_PARM *sparm)
{
  /* Finds the label ybar for pattern x that that is responsible for
     the most violated constraint for the margin rescaling
     formulation. It has to take into account the scoring function in
     sm, especially the weights sm.w, as well as the loss
     function. The weights in sm.w correspond to the features defined
     by psi() and range from index 1 to index sm->sizePsi. Most simple
     is the case of the zero/one loss function. For the zero/one loss,
     this function should return the highest scoring label ybar, if
     ybar is unequal y; if it is equal to the correct label y, then
     the function shall return the second highest scoring label. If
     the function cannot find a label, it shall return an empty label
     as recognized by the function empty_label(y). */
  LABEL ybar;
  int i,j;
  double *score,loss=0,scaling;
  MODEL svm_model;

  ybar.totdoc=x.totdoc;
  ybar.factor=(double *)my_malloc(sizeof(double)*x.totdoc);
  ybar._class=(double *)my_malloc(sizeof(double)*x.totdoc);
  score=ybar._class;
  svm_model=(*sm->svm_model); 
  scaling=0.5*x.scaling;

  for(i=0;i<x.totdoc;i++) {
    ybar.factor[i]=0;
    score[i]=classify_example(&svm_model,x.doc[i]);
  }
  for(i=0;i<x.totdoc;i++) {
    for(j=0;j<x.totdoc;j++) {
      if(y._class[i] > y._class[j]) {
	if((score[i] - score[j]) < 1) {
	  ybar.factor[j]+=scaling;
	  ybar.factor[i]-=scaling;
	  loss+=scaling; 
	}
	else {
	  ybar.factor[i]+=scaling;
	  ybar.factor[j]-=scaling;
	}
      }
    }
  }
  loss*=2.0;
  ybar.loss=loss;
  if(struct_verbosity >= 3) {
    SVECTOR *fy=psi(x,y,sm,sparm);
    SVECTOR *fybar=psi(x,ybar,sm,sparm);
    DOC *exy=create_example(0,0,1,1,fy);
    DOC *exybar=create_example(0,0,1,1,fybar);
    printf(" -> w*Psi(x,y_i)=%f, w*Psi(x,ybar)=%f, loss(y,ybar)=%f\n",
	   classify_example(sm->svm_model,exy),
	   classify_example(sm->svm_model,exybar),
	   loss);
    free_example(exy,1);
    free_example(exybar,1);
  }
  return(ybar);
}


int         empty_label(LABEL y)
{
  /* Returns true, if y is an empty label. An empty label might be
     returned by find_most_violated_constraint_???(x, y, sm) if there
     is no incorrect label that can be found for x, or if it is unable
     to label x at all */
  return(0);
}

SVECTOR     *psi(PATTERN x, LABEL y, STRUCTMODEL *sm,
		 STRUCT_LEARN_PARM *sparm)
{
  /* Returns a feature vector describing the match between pattern x
     and label y. The feature vector is returned as a list of
     SVECTOR's. Each SVECTOR is in a sparse representation of pairs
     <featurenumber:featurevalue>, where the last pair has
     featurenumber 0 as a terminator. Featurenumbers start with 1 and
     end with sizePsi. Featuresnumbers that are not specified default
     to value 0. As mentioned before, psi() actually returns a list of
     SVECTOR's. Each SVECTOR has a field 'factor' and 'next'. 'next'
     specifies the next element in the list, terminated by a NULL
     pointer. The list can be though of as a linear combination of
     vectors, where each vector is weighted by its 'factor'. This
     linear combination of feature vectors is multiplied with the
     learned (kernelized) weight vector to score label y for pattern
     x. Without kernels, there will be one weight in sm.w for each
     feature. Note that psi has to match
     find_most_violated_constraint_???(x, y, sm) and vice versa. In
     particular, find_most_violated_constraint_???(x, y, sm) finds
     that ybar!=y that maximizes psi(x,ybar,sm)*sm.w (where * is the
     inner vector product) and the appropriate function of the
     loss + margin/slack rescaling method. See that paper for details. */
  SVECTOR *fvec=NULL,*fvec2;
  double *sum;
  long i,totwords;

  /* The following special case speeds up computation for the linear
     kernel. The lines add the feature vectors for all examples into
     a single vector. This is more efficient for the linear kernel,
     but is invalid for all other kernels. */
  if(sm->svm_model->kernel_parm.kernel_type == LINEAR) {
    totwords=sparm->num_features;
    sum=(double *)my_malloc(sizeof(double)*(totwords+1));
    clear_nvector(sum,totwords);
    for(i=0;i<y.totdoc;i++) 
      add_vector_ns(sum,x.doc[i]->fvec,y.factor[i]);
    fvec=create_svector_n(sum,totwords,"",1);
    free(sum);
  }
  else { /* general kernel */
    for(i=0;i<y.totdoc;i++) {
      fvec2=copy_svector(x.doc[i]->fvec);
      fvec2->next=fvec;
      fvec2->factor=y.factor[i];
      fvec=fvec2;
    }
  }

  return(fvec);
}

double      loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM *sparm)
{
  /* loss for correct label y and predicted label ybar. The loss for
     y==ybar has to be zero. sparm->loss_function is set with the -l option. */
  double loss=1;

  if(sparm->loss_function == SWAPPEDPAIRS) {
    if(ybar.loss >= 0)
      loss=ybar.loss;            /* during training loss is provided by fmvc */
    else
      loss=swappedpairs(y,ybar); /* compute from y._class and ybar._class */
  }
  else if(sparm->loss_function == FRACSWAPPEDPAIRS) {
    if(ybar.loss >= 0)
      loss=ybar.loss;            /* during training loss is provided by fmvc */
    else
      loss=fracswappedpairs(y,ybar); /* compute from y._class and ybar._class */
  }
  else {
    /* Put your code for different loss functions here. But then
       find_most_violated_constraint_???(x, y, sm) has to return the
       highest scoring label with the largest loss. */
    printf("Unknown loss function type: %d\n",sparm->loss_function);
    exit(1);
  }

  return(loss);
}

int         finalize_iteration(double ceps, int cached_constraint,
			       SAMPLE sample, STRUCTMODEL *sm,
			       CONSTSET cset, double *alpha, 
			       STRUCT_LEARN_PARM *sparm)
{
  /* This function is called just before the end of each cutting plane iteration. ceps is the amount by which the most violated constraint found in the current iteration was violated. cached_constraint is true if the added constraint was constructed from the cache. If the return value is FALSE, then the algorithm is allowed to terminate. If it is TRUE, the algorithm will keep iterating even if the desired precision sparm->epsilon is already reached. */
  return(0);
}

void        print_struct_learning_stats(SAMPLE sample, STRUCTMODEL *sm,
					CONSTSET cset, double *alpha, 
					STRUCT_LEARN_PARM *sparm)
{
  /* This function is called after training and allows final touches to
     the model sm. But primarly it allows computing and printing any
     kind of statistic (e.g. training error) you might want. */
  MODEL *model=sm->svm_model;

  /* Replace SV with single weight vector */
  if(model->kernel_parm.kernel_type == LINEAR) {
    if(struct_verbosity>=1) {
      printf("Compacting linear model..."); fflush(stdout);
    }
    sm->svm_model=compact_linear_model(model);
    sm->w=sm->svm_model->lin_weights; /* short cut to weight vector */
    free_model(model,1);
    if(struct_verbosity>=1) {
      printf("done\n"); fflush(stdout);
    }
  }  
}

void        print_struct_testing_stats(SAMPLE sample, STRUCTMODEL *sm,
				       STRUCT_LEARN_PARM *sparm, 
				       STRUCT_TEST_STATS *teststats)
{
  /* This function is called after making all test predictions in
     svm_struct_classify and allows computing and printing any kind of
     evaluation (e.g. precision/recall) you might want. You can use
     the function eval_prediction to accumulate the necessary
     statistics for each prediction. */

  printf("NOTE: The loss reported above is the fraction of swapped pairs averaged over\n");
  printf("      all rankings. The zero/one-error is fraction of perfectly correct\n");
  printf("      rankings!\n");
  teststats->fracswappedpairs/=sample.n;
  printf("Total Num Swappedpairs  : %6.0f\n",teststats->swappedpairs);
  printf("Avg Swappedpairs Percent: %6.2f\n",100.0*teststats->fracswappedpairs);
}

void        eval_prediction(long exnum, EXAMPLE ex, LABEL ypred, 
			    STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, 
			    STRUCT_TEST_STATS *teststats)
{
  /* This function allows you to accumlate statistic for how well the
     prediction matches the labeled example. It is called from
     svm_struct_classify. See also the function
     print_struct_testing_stats. */
  if(exnum == 0) { /* this is the first time the function is
		      called. So initialize the teststats */
    teststats->swappedpairs=0;
    teststats->fracswappedpairs=0;
  }
  teststats->swappedpairs+=swappedpairs(ex.y,ypred);
  teststats->fracswappedpairs+=fracswappedpairs(ex.y,ypred);
}

void        write_struct_model(char *file, STRUCTMODEL *sm, 
			       STRUCT_LEARN_PARM *sparm)
{
  /* Writes structural model sm to file file. */
  /* Store model in normal svm-light format */
  write_model(file,sm->svm_model);
}

STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm)
{
  /* Reads structural model sm from file file. This function is used
     only in the prediction module, not in the learning module. */
  STRUCTMODEL sm;
  
  sm.svm_model=read_model(file);
  sparm->loss_function=FRACSWAPPEDPAIRS;
  sparm->num_features=sm.svm_model->totwords;
  sm.w=sm.svm_model->lin_weights;
  sm.sizePsi=sm.svm_model->totwords;
  return(sm);
}

void        write_label(FILE *fp, LABEL y)
{
  /* Writes label y to file handle fp. */
  int i;
  for(i=0;i<y.totdoc;i++) {
    fprintf(fp,"%.8f\n",y._class[i]);
  }
}

void        free_pattern(PATTERN x) {
  /* Frees the memory of x. */
  int i;
  for(i=0;i<x.totdoc;i++) 
    free_example(x.doc[i],1);
  free(x.doc);
}

void        free_label(LABEL y) {
  /* Frees the memory of y. */
  free(y._class);
  if(y.factor)
    free(y.factor);
}

void        free_struct_model(STRUCTMODEL sm) 
{
  /* Frees the memory of model. */

  /* if(sm.w) free(sm.w); */ /* this is free'd in free_model */
  if(sm.svm_model) free_model(sm.svm_model,1);
  /* add free calls for user defined data here */
}

void        free_struct_sample(SAMPLE s)
{
  /* Frees the memory of sample s. */
  int i;
  for(i=0;i<s.n;i++) { 
    free_pattern(s.examples[i].x);
    free_label(s.examples[i].y);
  }
  free(s.examples);
}

void        print_struct_help()
{
  /* Prints a help text that is appended to the common help text of
     svm_struct_learn. */
  printf("\nThe following loss functions can be selected with the -l option:\n");
  printf("    %2d  Total number of swapped pairs summed over all queries.\n",SWAPPEDPAIRS);
  printf("    %2d  Fraction of swapped pairs averaged over all queries.\n\n",FRACSWAPPEDPAIRS);
  printf("NOTE: SVM-light in '-z p' mode and SVM-rank with loss %d are equivalent for\n",SWAPPEDPAIRS);
  printf("      c_light = c_rank/n, where n is the number of training rankings (i.e. \n");
  printf("      queries).\n\n");
  printf("The algorithms implemented in SVM-perf are described in:\n");
  printf("- T. Joachims, A Support Vector Method for Multivariate Performance Measures,\n");
  printf("  Proceedings of the International Conference on Machine Learning (ICML), 2005.\n");
  printf("- T. Joachims, Training Linear SVMs in Linear Time, Proceedings of the \n");
  printf("  ACM Conference on Knowledge Discovery and Data Mining (KDD), 2006.\n");
  printf("  -> Papers are available at http://www.joachims.org/\n\n");
}

void         parse_struct_parameters(STRUCT_LEARN_PARM *sparm)
{
  /* Parses the command line parameters that start with -- */
  int i;

  /* set number of features to -1, indicating that it will be computed
     in init_struct_model() */
  sparm->num_features=-1;

  for(i=0;(i<sparm->custom_argc) && ((sparm->custom_argv[i])[0] == '-');i++) {
    switch ((sparm->custom_argv[i])[2]) 
      { 
      default: printf("\nUnrecognized option %s!\n\n",sparm->custom_argv[i]);
	       exit(0);
      }
  }
}

void        print_struct_help_classify()
{
  /* Prints a help text that is appended to the common help text of
     svm_struct_classify. */
}

void         parse_struct_parameters_classify(STRUCT_LEARN_PARM *sparm)
{
  /* Parses the command line parameters that start with -- for the
     classification module */
  int i;

  for(i=0;(i<sparm->custom_argc) && ((sparm->custom_argv[i])[0] == '-');i++) {
    switch ((sparm->custom_argv[i])[2]) 
      { 
      /* case 'x': i++; strcpy(xvalue,sparm->custom_argv[i]); break; */
      default: printf("\nUnrecognized option %s!\n\n",sparm->custom_argv[i]);
	       exit(0);
      }
  }
}


/*------- Performance measures --------*/

double swappedpairs(LABEL y, LABEL ybar)
{
  /* Returns number of swapped pairs for prediction vectors that
     encode the number of misranked examples for each particular
     example. */
  /* WARNING: y needs to be the correct ranking, and ybar the prediction. */
  int i,j;
  double sum=0;
  for(i=0;i<y.totdoc;i++) {
    for(j=0;j<y.totdoc;j++) {
      if((y._class[i]>y._class[j]) && (ybar._class[i]<=ybar._class[j])) {
	sum++;
      }
    }
  }
  return(sum);
}

double fracswappedpairs(LABEL y, LABEL ybar)
{
  /* Returns fraction of swapped pairs for prediction vectors that
     encode the number of misranked examples for each particular
     example. */
  /* WARNING: y needs to be the correct ranking, and ybar the prediction. */
  int i,j;
  double sum=0,totpairs=0;
  for(i=0;i<y.totdoc;i++) {
    for(j=0;j<y.totdoc;j++) {
      if(y._class[i]>y._class[j]) {
	totpairs++;
	if(ybar._class[i]<=ybar._class[j]) {
	  sum++;
	}
      }
    }
  }
  if(totpairs)
    return((double)sum/(double)totpairs);
  else
    return(0);
}

