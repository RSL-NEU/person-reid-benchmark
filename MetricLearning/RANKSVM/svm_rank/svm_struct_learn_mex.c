/***********************************************************************/
/*                                                                     */
/*   svm_struct_main.c                                                 */
/*                                                                     */
/*   Command line interface to the alignment learning module of the    */
/*   Support Vector Machine.                                           */
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 03.07.04                                                    */
/*                                                                     */
/*   Copyright (c) 2004  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/

#ifdef __cplusplus
extern "C" {
#endif
    
#include "svm_light/svm_common.h"
#include "svm_light/svm_learn.h"
    
#ifdef __cplusplus
}
#endif

# include "svm_struct/svm_struct_learn.h"
# include "svm_struct/svm_struct_common.h"
# include "svm_struct_api.h"
#include "mex.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

void read_input_parameters(int, char **,
        long *, long *,
        STRUCT_LEARN_PARM *, LEARN_PARM *, KERNEL_PARM *,
        int *);

void arg_split(char *string, int *argc, char ***argv);
void load_samples(SAMPLE *sample, double *train_features, double *train_image_num_per_person, int *cum_train_image_num, int train_size, int dimension, int* dnum);
void partial_read_struct_examples(SAMPLE *sample, STRUCT_LEARN_PARM *sparm);
mxArray* write_model_to_mexArray(STRUCTMODEL *smodel, int dimension);
/** ------------------------------------------------------------------
 ** @brief MEX entry point
 **/

void mexFunction(int nout, mxArray **out, int nin, mxArray const **in)
{
    SAMPLE sample;  /* training sample */
    LEARN_PARM learn_parm;
    KERNEL_PARM kernel_parm;
    STRUCT_LEARN_PARM struct_parm;
    STRUCTMODEL structmodel;
    int alg_type;
    
    PATTERN *x;
    LABEL *y;
    WORD* words;
    int wpos;
    
    enum {IN_ARGS=0, IN_SPARM} ;
    enum {OUT_W=0} ;
    
    char arg [1024 + 1] ;
    int argc ;
    char ** argv ;
    
    mxArray const * sparm_mxArray;
    mxArray const * train_features_mxArray;
    mxArray const * train_image_num_per_person_mxArray;
    
    double *train_features;
    double *train_image_num_per_person;
    int *cum_train_image_num;
    int train_size, dimension, total_image_num, dnum;
    int i, tmp;
    mxArray * model_array;
    
    if (nin != 2) {
        mexErrMsgTxt("Two arguments required") ;
    }
    
    /* Parse ARGS  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
    
    mxGetString(in[IN_ARGS], arg, sizeof(arg) / sizeof(char)) ;
    arg_split (arg, &argc, &argv) ;
    
    svm_struct_learn_api_init(argc+1, argv-1) ;
    
    read_input_parameters (argc+1,argv-1,
            &verbosity, &struct_verbosity,
            &struct_parm, &learn_parm,
            &kernel_parm, &alg_type ) ;
    
    /* Parse SPARM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
    sparm_mxArray = in [IN_SPARM] ;
    
    train_features_mxArray = mxGetField(sparm_mxArray, 0, "TrainFeatures");
    if (! train_features_mxArray ||
            ! mxIsDouble(train_features_mxArray)) {
        mexErrMsgTxt("SPARM.PATTERNS must be a double array");
    }
    dimension = mxGetM(train_features_mxArray);
    total_image_num = mxGetN(train_features_mxArray);
    mexPrintf("Read in training features %d dimension, total %d\n", dimension, total_image_num);
    
    train_features = (double *)my_malloc(sizeof(double) * dimension * total_image_num);
    memcpy(train_features, mxGetPr(train_features_mxArray), sizeof(double) * dimension * total_image_num);
    
    train_image_num_per_person_mxArray = mxGetField(sparm_mxArray, 0, "TrainImagesNumPerPerson");
    train_size = mxGetNumberOfElements(train_image_num_per_person_mxArray);
    train_image_num_per_person = (double *)my_malloc(sizeof(double) * train_size);
    memcpy(train_image_num_per_person, mxGetPr(train_image_num_per_person_mxArray), sizeof(double) * train_size);

    cum_train_image_num = (double *)my_malloc(sizeof(int) * train_size);
    tmp = 0;
    for(i = 0; i < train_size; i++){
        cum_train_image_num[i] = tmp;
        tmp += (int)train_image_num_per_person[i];
    }
    assert(tmp == total_image_num);

    sample.n = train_size;
    sample.examples = (EXAMPLE *) my_malloc (sizeof(EXAMPLE) * train_size);
    dnum = 0;

    load_samples(&sample, train_features, train_image_num_per_person, cum_train_image_num, train_size, dimension, &dnum);

    free(train_features);
    free(train_image_num_per_person);
    free(cum_train_image_num);

    partial_read_struct_examples(&sample, &struct_parm);
 
    if (struct_verbosity >= 1) {
        mexPrintf("There are %d training examples, algorithm type %d\n", dnum, alg_type);
    }

    /* Learning  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
    switch (alg_type) {
        case 0:
            svm_learn_struct(sample,&struct_parm,&learn_parm,&kernel_parm,&structmodel,NSLACK_ALG) ;
            break ;
        case 1:
            svm_learn_struct(sample,&struct_parm,&learn_parm,&kernel_parm,&structmodel,NSLACK_SHRINK_ALG);
            break ;
        case 2:
            svm_learn_struct_joint(sample,&struct_parm,&learn_parm,&kernel_parm,&structmodel,ONESLACK_PRIMAL_ALG);
            break ;
        case 3:
            svm_learn_struct_joint(sample,&struct_parm,&learn_parm,&kernel_parm,&structmodel,ONESLACK_DUAL_ALG);
            break ;
        case 4:
            svm_learn_struct_joint(sample,&struct_parm,&learn_parm,&kernel_parm,&structmodel,ONESLACK_DUAL_CACHE_ALG);
            break  ;
        case 9:
            svm_learn_struct_joint_custom(sample,&struct_parm,&learn_parm,&kernel_parm,&structmodel);
            break ;
        default:
            mexErrMsgTxt("Unknown algorithm type") ;
    }
    
    /* Write output  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
    
//    model_array = newMxArrayEncapsulatingSmodel (&structmodel) ;
	out[OUT_W] = write_model_to_mexArray(&structmodel, dimension);
//    out[OUT_W] = mxDuplicateArray (model_array) ;
//    destroyMxArrayEncapsulatingSmodel (model_array) ;

    //printf("after destroy all array\n");
    //mexCallMATLAB(0, NULL, 0, NULL, "keyboard");
    free_struct_sample (sample) ;
    free_struct_model (structmodel) ;
    svm_struct_learn_api_exit () ;
}

mxArray* write_model_to_mexArray(STRUCTMODEL *smodel, int dimension){
	int j, i, sv_num;
	SVECTOR *v;
	MODEL *model;
	mxArray *x;
	double *model_data;

	model = smodel->svm_model;
	x = mxCreateDoubleMatrix(dimension, 1, mxREAL);
	model_data = (double *)my_malloc(sizeof(double) * dimension);
	for(i = 0; i < dimension; i++){
		model_data[i] = 0;
	}
	sv_num=1;
	for(i=1;i<model->sv_num;i++) {
		for(v=model->supvec[i]->fvec;v;v=v->next)
			sv_num++;
	}
	for(i=1;i<model->sv_num;i++) {
		for(v=model->supvec[i]->fvec;v;v=v->next) {
			for (j=0; (v->words[j]).wnum; j++) {
				assert(dimension >= v->words[j].wnum);
				model_data[(v->words[j]).wnum - 1] = (double)(v->words[j]).weight;
			}
			/* NOTE: this could be made more efficient by summing the
			alpha's of identical vectors before writing them to the
			file. */
		}
	}
	memcpy(mxGetPr(x), model_data, sizeof(double) * dimension);
	free(model_data);
	return x;
}

/** ------------------------------------------------------------------
 ** @brief Parse argument string
 **/

void read_input_parameters (int argc,char *argv[],
        long *verbosity,long *struct_verbosity,
        STRUCT_LEARN_PARM *struct_parm,
        LEARN_PARM *learn_parm, KERNEL_PARM *kernel_parm,
        int *alg_type)
{
    long i ;
    
    (*alg_type)=DEFAULT_ALG_TYPE;
    
    /* SVM struct options */
    (*struct_verbosity)=1;
    
    struct_parm->C=-0.01;
    struct_parm->slack_norm=1;
    struct_parm->epsilon=DEFAULT_EPS;
    struct_parm->custom_argc=0;
    struct_parm->loss_function=DEFAULT_LOSS_FCT;
    struct_parm->loss_type=DEFAULT_RESCALING;
    struct_parm->newconstretrain=100;
    struct_parm->ccache_size=5;
    struct_parm->batch_size=100;
    
    /* SVM light options */
    (*verbosity)=0;
    
    strcpy (learn_parm->predfile, "trans_predictions");
    strcpy (learn_parm->alphafile, "");
    learn_parm->biased_hyperplane=1;
    learn_parm->remove_inconsistent=0;
    learn_parm->skip_final_opt_check=0;
    learn_parm->svm_maxqpsize=10;
    learn_parm->svm_newvarsinqp=0;
    learn_parm->svm_iter_to_shrink=-9999;
    learn_parm->maxiter=100000;
    learn_parm->kernel_cache_size=40;
    learn_parm->svm_c=99999999;  /* overridden by struct_parm->C */
    learn_parm->eps=0.001;       /* overridden by struct_parm->epsilon */
    learn_parm->transduction_posratio=-1.0;
    learn_parm->svm_costratio=1.0;
    learn_parm->svm_costratio_unlab=1.0;
    learn_parm->svm_unlabbound=1E-5;
    learn_parm->epsilon_crit=0.001;
    learn_parm->epsilon_a=1E-10;  /* changed from 1e-15 */
    learn_parm->compute_loo=0;
    learn_parm->rho=1.0;
    learn_parm->xa_depth=0;
    
    kernel_parm->kernel_type=0;
    kernel_parm->poly_degree=3;
    kernel_parm->rbf_gamma=1.0;
    kernel_parm->coef_lin=1;
    kernel_parm->coef_const=1;
    strcpy (kernel_parm->custom,"empty");
    
    /* Parse -x options, delegat --x ones */
    for(i=1;(i<argc) && ((argv[i])[0] == '-');i++) {
        switch ((argv[i])[1])
        {
            case 'a': i++; strcpy(learn_parm->alphafile,argv[i]); break;
            case 'c': i++; struct_parm->C=atof(argv[i]); break;
            case 'p': i++; struct_parm->slack_norm=atol(argv[i]); break;
            case 'e': i++; struct_parm->epsilon=atof(argv[i]); break;
            case 'k': i++; struct_parm->newconstretrain=atol(argv[i]); break;
            case 'h': i++; learn_parm->svm_iter_to_shrink=atol(argv[i]); break;
            case '#': i++; learn_parm->maxiter=atol(argv[i]); break;
            case 'm': i++; learn_parm->kernel_cache_size=atol(argv[i]); break;
            case 'w': i++; (*alg_type)=atol(argv[i]); break;
            case 'o': i++; struct_parm->loss_type=atol(argv[i]); break;
            case 'n': i++; learn_parm->svm_newvarsinqp=atol(argv[i]); break;
            case 'q': i++; learn_parm->svm_maxqpsize=atol(argv[i]); break;
            case 'l': i++; struct_parm->loss_function=atol(argv[i]); break;
            case 'f': i++; struct_parm->ccache_size=atol(argv[i]); break;
            case 'b': i++; struct_parm->batch_size=atof(argv[i]); break;
            case 't': i++; kernel_parm->kernel_type=atol(argv[i]); break;
            case 'd': i++; kernel_parm->poly_degree=atol(argv[i]); break;
            case 'g': i++; kernel_parm->rbf_gamma=atof(argv[i]); break;
            case 's': i++; kernel_parm->coef_lin=atof(argv[i]); break;
            case 'r': i++; kernel_parm->coef_const=atof(argv[i]); break;
            case 'u': i++; strcpy(kernel_parm->custom,argv[i]); break;
            case 'v': i++; (*struct_verbosity)=atol(argv[i]); break;
            case 'y': i++; (*verbosity)=atol(argv[i]); break;
            case '-':
                strcpy(struct_parm->custom_argv[struct_parm->custom_argc++],argv[i]);
                i++;
                strcpy(struct_parm->custom_argv[struct_parm->custom_argc++],argv[i]);
                break;
            default:
            {
                char msg [1024+1] ;
#ifndef WIN
                snprintf(msg, sizeof(msg)/sizeof(char),
                        "Unrecognized option '%s'",argv[i]) ;
#else
                sprintf(msg, sizeof(msg)/sizeof(char),
                        "Unrecognized option '%s'",argv[i]) ;
#endif
                mexErrMsgTxt(msg) ;
            }
        }
    }
    
    /* whatever is left is an error */
    if (i < argc) {
        char msg [1024+1] ;
#ifndef WIN
        snprintf(msg, sizeof(msg)/sizeof(char),
                "Unrecognized argument '%s'", argv[i]) ;
#else
        sprintf(msg, sizeof(msg)/sizeof(char),
                "Unrecognized argument '%s'", argv[i]) ;
#endif
        mexErrMsgTxt(msg) ;
    }
    
    /* Check parameter validity */
    if(learn_parm->svm_iter_to_shrink == -9999) {
        learn_parm->svm_iter_to_shrink=100;
    }
    
    if((learn_parm->skip_final_opt_check)
    && (kernel_parm->kernel_type == LINEAR)) {
        mexWarnMsgTxt("It does not make sense to skip the final optimality check for linear kernels.");
        learn_parm->skip_final_opt_check=0;
    }
    if((learn_parm->skip_final_opt_check)
    && (learn_parm->remove_inconsistent)) {
        mexErrMsgTxt("It is necessary to do the final optimality check when removing inconsistent examples.");
    }
    if((learn_parm->svm_maxqpsize<2)) {
        char msg [1025] ;
#ifndef WIN
        snprintf(msg, sizeof(msg)/sizeof(char),
                "Maximum size of QP-subproblems not in valid range: %ld [2..]",learn_parm->svm_maxqpsize) ;
#else
        sprintf(msg, sizeof(msg)/sizeof(char),
                "Maximum size of QP-subproblems not in valid range: %ld [2..]",learn_parm->svm_maxqpsize) ;
#endif
        mexErrMsgTxt(msg) ;
    }
    if((learn_parm->svm_maxqpsize<learn_parm->svm_newvarsinqp)) {
        char msg [1025] ;
#ifndef WIN
        snprintf(msg, sizeof(msg)/sizeof(char),
                "Maximum size of QP-subproblems [%ld] must be larger than the number of"
                " new variables [%ld] entering the working set in each iteration.",
                learn_parm->svm_maxqpsize, learn_parm->svm_newvarsinqp) ;
#else
        sprintf(msg, sizeof(msg)/sizeof(char),
                "Maximum size of QP-subproblems [%ld] must be larger than the number of"
                " new variables [%ld] entering the working set in each iteration.",
                learn_parm->svm_maxqpsize, learn_parm->svm_newvarsinqp) ;
#endif
        mexErrMsgTxt(msg) ;
    }
    if(learn_parm->svm_iter_to_shrink<1) {
        char msg [1025] ;
#ifndef WIN
        snprintf(msg, sizeof(msg)/sizeof(char),
                "Maximum number of iterations for shrinking not in valid range: %ld [1,..]",
                learn_parm->svm_iter_to_shrink);
#else
        sprintf(msg, sizeof(msg)/sizeof(char),
                "Maximum number of iterations for shrinking not in valid range: %ld [1,..]",
                learn_parm->svm_iter_to_shrink);
#endif
        mexErrMsgTxt(msg) ;
    }
    if(struct_parm->C<0) {
        mexErrMsgTxt("You have to specify a value for the parameter '-c' (C>0)!");
    }
    if(((*alg_type) < 0) || (((*alg_type) > 5) && ((*alg_type) != 9))) {
        mexErrMsgTxt("Algorithm type must be either '0', '1', '2', '3', '4', or '9'!");
    }
    if(learn_parm->transduction_posratio>1) {
        mexErrMsgTxt("The fraction of unlabeled examples to classify as positives must "
                "be less than 1.0 !!!");
    }
    if(learn_parm->svm_costratio<=0) {
        mexErrMsgTxt("The COSTRATIO parameter must be greater than zero!");
    }
    if(struct_parm->epsilon<=0) {
        mexErrMsgTxt("The epsilon parameter must be greater than zero!");
    }
    if((struct_parm->ccache_size<=0) && ((*alg_type) == 4)) {
        mexErrMsgTxt("The cache size must be at least 1!");
    }
    if(((struct_parm->batch_size<=0) || (struct_parm->batch_size>100))
    && ((*alg_type) == 4)) {
        mexErrMsgTxt("The batch size must be in the interval ]0,100]!");
    }
    if((struct_parm->slack_norm<1) || (struct_parm->slack_norm>2)) {
        mexErrMsgTxt("The norm of the slacks must be either 1 (L1-norm) or 2 (L2-norm)!");
    }
    if((struct_parm->loss_type != SLACK_RESCALING)
    && (struct_parm->loss_type != MARGIN_RESCALING)) {
        mexErrMsgTxt("The loss type must be either 1 (slack rescaling) or 2 (margin rescaling)!");
    }
    if(learn_parm->rho<0) {
        mexErrMsgTxt("The parameter rho for xi/alpha-estimates and leave-one-out pruning must"
                " be greater than zero (typically 1.0 or 2.0, see T. Joachims, Estimating the"
                " Generalization Performance of an SVM Efficiently, ICML, 2000.)!");
    }
    if((learn_parm->xa_depth<0) || (learn_parm->xa_depth>100)) {
        mexErrMsgTxt("The parameter depth for ext. xi/alpha-estimates must be in [0..100] (zero"
                "for switching to the conventional xa/estimates described in T. Joachims,"
                "Estimating the Generalization Performance of an SVM Efficiently, ICML, 2000.)") ;
    }
    
    parse_struct_parameters (struct_parm) ;
}

void arg_split (char *string, int *argc, char ***argv)
{
    size_t size;
    char *d, *p;
    
    for (size = 1, p = string; *p; p++) {
        if (isspace((int) *p)) {
            size++;
        }
    }
    size++;			/* leave space for final NULL pointer. */
    
    *argv = (char **) my_malloc(((size * sizeof(char *)) + (p - string) + 1));
    
    for (*argc = 0, p = string, d = ((char *) *argv) + size*sizeof(char *);
    *p != 0; ) {
        (*argv)[*argc] = NULL;
        while (*p && isspace((int) *p)) p++;
        if (*argc == 0 && *p == '#') {
            break;
        }
        if (*p) {
            char *s = p;
            (*argv)[(*argc)++] = d;
            while (*p && !isspace((int) *p)) p++;
            memcpy(d, s, p-s);
            d += p-s;
            *d++ = 0;
            while (*p && isspace((int) *p)) p++;
        }
    }
}

void write_sample(int m, double* f1, double* f2, PATTERN* x, LABEL* y, int rank, int dimension, int *dnum){
    //rank - sample rank, positive->2, negative->1
    WORD* words;
    int wpos, d;
    double abs_feature;

    x->totdoc++;
    y->totdoc++;
    y->_class[y->totdoc - 1] = rank;
    words = (WORD *)my_malloc(sizeof(WORD) * dimension);
    wpos = 0;
    for(d = 0; d < dimension; d++){
        abs_feature = fabs(*(f1 + d) - *(f2+ d));
        if(abs_feature != 0){
            words[wpos].wnum = d + 1;
            words[wpos].weight = (FVAL)abs_feature;
            wpos++;
        }
    }
    words[wpos].wnum = 0;
    x->doc[x->totdoc -1 ] = create_example((*dnum)++, m + 1, 0, 1, create_svector(words, "", 1.0));
    free(words);
}

void load_samples(SAMPLE *sample, double *train_features, double *train_image_num_per_person, int *cum_train_image_num, int train_size, int dimension, int* dnum){
    PATTERN *x;
    LABEL *y;
    WORD *words;
    int wpos;

    double abs_feature;
    double *f1, *f2;
    int img_num_curr_person, img_num_cand_person, constraints_per_person, total_image_num = 0;
    int i, j, m, n, d;

    for(i = 0; i < train_size; i++){
        total_image_num += train_image_num_per_person[i];
    }

    for(i = 0; i < train_size; i++){
        img_num_curr_person = (int)train_image_num_per_person[i];
        constraints_per_person = img_num_curr_person * (img_num_curr_person - 1) / 2 + img_num_curr_person * (total_image_num - img_num_curr_person);
        
        x = &(sample->examples[i].x);
        y = &(sample->examples[i].y);
        x->doc = (DOC**)my_malloc(sizeof(DOC*) * constraints_per_person);
        x->totdoc = 0;

        y->factor = (double*)my_malloc(sizeof(double) * constraints_per_person);
        y->_class = (double*)my_malloc(sizeof(double) * constraints_per_person);
        y->loss = 0;
        y->totdoc = 0;

        for(j = 0; j < train_size; j++){
            img_num_cand_person = (int)train_image_num_per_person[j];
            for(m = 0; m < img_num_curr_person; m++){
                if( i == j){ // positive samples
                    for(n = m + 1; n < img_num_curr_person; n++){
                        f1 = train_features + dimension * (cum_train_image_num[i] + m);
                        f2 = train_features + dimension * (cum_train_image_num[i] + n);
                        write_sample(m, f1, f2, x, y, 2, dimension, dnum);
                    }
                }else{
                    for(n = 0; n < img_num_cand_person; n++){
                        f1 = train_features + dimension * (cum_train_image_num[i] + m);
                        f2 = train_features + dimension * (cum_train_image_num[j] + n);
                        write_sample(m, f1, f2, x, y, 1, dimension, dnum);
                    }
                }

            }
        }
        assert(x->totdoc == constraints_per_person);
        assert(y->totdoc == constraints_per_person);
    }
}

void partial_read_struct_examples(SAMPLE *sample, STRUCT_LEARN_PARM *sparm){
	long k, i, j, sumtotpairs, totpairs;
	PATTERN  *x;
	LABEL    *y;
	//SAMPLE sample = *testsample;
	if(sparm->num_features > 0) 
		for(k=0;k<sample->n;k++) 
			for(i=0;i<sample->examples[k].x.totdoc;i++)
				for(j=0;sample->examples[k].x.doc[i]->fvec->words[j].wnum;j++) 
					if(sample->examples[k].x.doc[i]->fvec->words[j].wnum>sparm->num_features) {
						sample->examples[k].x.doc[i]->fvec->words[j].wnum=0;
						sample->examples[k].x.doc[i]->fvec->words[j].weight=0;
					}

	/* add label factors for easy computation of feature vectors */
	sumtotpairs=0;
	for(k=0;k<sample->n;k++) {
		x=&(sample->examples[k].x);
		y=&(sample->examples[k].y);
		//y->factor=(double *)my_malloc(sizeof(double)*y->totdoc);
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
		sparm->epsilon*=(double)sumtotpairs/(double)sample->n; /* adjusting eps */
		for(k=0;k<sample->n;k++) {
			x=&(sample->examples[k].x);
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
	for(k=0;k<sample->n;k++) {
		x=&(sample->examples[k].x);
		y=&(sample->examples[k].y);
		for(i=0;i<y->totdoc;i++) 
			y->factor[i]*=x->scaling;
	}
	//*testsample = sample;
};
