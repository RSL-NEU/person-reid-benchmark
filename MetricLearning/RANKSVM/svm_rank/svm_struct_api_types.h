/***********************************************************************/
/*                                                                     */
/*   svm_struct_api_types.h (instantiated for SVM-rank)                */
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

#ifndef svm_struct_api_types
#define svm_struct_api_types

# include "svm_light/svm_common.h"
# include "svm_light/svm_learn.h"

# ifndef WIN
# include "strings.h"
# else
# include <string.h>
# endif

#ifndef WIN
#define inline_comm __inline__
#else
#define inline_comm __inline
#endif

# define INST_NAME          "SVM-rank"
# define INST_VERSION       "V1.00"
# define INST_VERSION_DATE  "15.03.2009"

/* Identifiers for loss functions */
#define SWAPPEDPAIRS     1
#define FRACSWAPPEDPAIRS 2

/* default precision for solving the optimization problem */
# define DEFAULT_EPS         0.001 
/* default loss rescaling method: 1=slack_rescaling, 2=margin_rescaling */
# define DEFAULT_RESCALING   2
/* default loss function:    */
# define DEFAULT_LOSS_FCT    SWAPPEDPAIRS
/* default optimization algorithm to use: */
# define DEFAULT_ALG_TYPE    3
/* store Psi(x,y) once instead of recomputing it every time: */
# define USE_FYCACHE         1
/* decide whether to evaluate sum before storing vectors in constraint
   cache: 
   0 = NO, 
   1 = YES (best, if sparse vectors and long vector lists), 
   2 = YES (best, if short vector lists),
   3 = YES (best, if dense vectors and long vector lists) */
# define COMPACT_CACHED_VECTORS 2
/* minimum absolute value below which values in sparse vectors are
   rounded to zero. Values are stored in the FVAL type defined in svm_common.h 
   RECOMMENDATION: assuming you use FVAL=float, use 
     10E-15 if COMPACT_CACHED_VECTORS is 1 
     10E-10 if COMPACT_CACHED_VECTORS is 2 or 3 
*/
# define COMPACT_ROUNDING_THRESH 10E-15


typedef struct pattern {
  /* this defines the x-part of a training example, e.g. the structure
     for storing a natural language sentence in NLP parsing */
  DOC    **doc;     /* set of example vectors */
  double scaling;   /* factor to scale loss and feature vectors by to
		       get a more sensible range for the stopping
		       criterion and for C */
  int totdoc;       /* size of set */
} PATTERN;

typedef struct label {
  /* this defines the y-part (the label) of a training example,
     e.g. the parse tree of the corresponding sentence. */
  double *factor;   /* vector of document factors */
  double loss;      /* loss of this label (used during training) */
  double *_class;    /* vector of scores that imply ranking */
  int totdoc;       /* size of set */
} LABEL;

typedef struct structmodel {
  double *w;          /* pointer to the learned weights */
  MODEL  *svm_model;  /* the learned SVM model */
  long   sizePsi;     /* maximum number of weights in w */
  /* other information that is needed for the stuctural model can be
     added here, e.g. the grammar rules for NLP parsing */
} STRUCTMODEL;

typedef struct struct_learn_parm {
  double epsilon;              /* precision for which to solve
				  quadratic program */
  double newconstretrain;      /* number of new constraints to
				  accumulate before recomputing the QP
				  solution */
  int    ccache_size;          /* maximum number of constraints to
				  cache for each example (used in w=4
				  algorithm) */
  double batch_size;           /* size of the mini batches in percent
				  of training set size (used in w=4
				  algorithm) */
  double C;                    /* trade-off between margin and loss */
  char   custom_argv[20][300]; /* string set with the -u command line option */
  int    custom_argc;          /* number of -u command line options */
  int    slack_norm;           /* norm to use in objective function
                                  for slack variables; 1 -> L1-norm, 
				  2 -> L2-norm */
  int    loss_type;            /* selected loss rescaline type from -r
				  command line option. Select between
				  slack rescaling (1) and margin
				  rescaling (2) */
  int    loss_function;        /* select between different loss
				  functions via -l command line
				  option */
  /* further parameters that are passed to init_struct_model() */
  int num_features;
} STRUCT_LEARN_PARM;

typedef struct struct_test_stats {
  /* you can add variables for keeping statistics when evaluating the
     test predictions in svm_struct_classify. This can be used in the
     function eval_prediction and print_struct_testing_stats. */
  double swappedpairs;
  double fracswappedpairs;
} STRUCT_TEST_STATS;

typedef struct struct_id_score {
  int id;
  double score;
  double tiebreak;
} STRUCT_ID_SCORE;

#endif