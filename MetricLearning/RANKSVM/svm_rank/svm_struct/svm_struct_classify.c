/***********************************************************************/
/*                                                                     */
/*   svm_struct_classify.c                                             */
/*                                                                     */
/*   Classification module of SVM-struct.                              */
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
/************************************************************************/

#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif
#include "../svm_light/svm_common.h"
#ifdef __cplusplus
}
#endif
#include "../svm_struct_api.h"
#include "svm_struct_common.h"

char testfile[200];
char modelfile[200];
char predictionsfile[200];

void read_input_parameters(int, char **, char *, char *, char *, 
			   STRUCT_LEARN_PARM *, long*, long *);
void print_help(void);


int main (int argc, char* argv[])
{
  long correct=0,incorrect=0,no_accuracy=0;
  long i;
  double t1,runtime=0;
  double avgloss=0,l;
  FILE *predfl;
  STRUCTMODEL model; 
  STRUCT_LEARN_PARM sparm;
  STRUCT_TEST_STATS teststats;
  SAMPLE testsample;
  LABEL y;

  svm_struct_classify_api_init(argc,argv);

  read_input_parameters(argc,argv,testfile,modelfile,predictionsfile,&sparm,
			&verbosity,&struct_verbosity);

  if(struct_verbosity>=1) {
    printf("Reading model..."); fflush(stdout);
  }
  model=read_struct_model(modelfile,&sparm);
  if(struct_verbosity>=1) {
    fprintf(stdout, "done.\n");
  }

  if(model.svm_model->kernel_parm.kernel_type == LINEAR) { /* linear kernel */
    /* compute weight vector */
    add_weight_vector_to_linear_model(model.svm_model);
    model.w=model.svm_model->lin_weights;
  }
  
  if(struct_verbosity>=1) {
    printf("Reading test examples..."); fflush(stdout);
  }
  testsample=read_struct_examples(testfile,&sparm);
  if(struct_verbosity>=1) {
    printf("done.\n"); fflush(stdout);
  }

  if(struct_verbosity>=1) {
    printf("Classifying test examples..."); fflush(stdout);
  }

  if ((predfl = fopen (predictionsfile, "w")) == NULL)
  { perror (predictionsfile); exit (1); }

  for(i=0;i<testsample.n;i++) {
    t1=get_runtime();
    y=classify_struct_example(testsample.examples[i].x,&model,&sparm);
    runtime+=(get_runtime()-t1);

    write_label(predfl,y);
    l=loss(testsample.examples[i].y,y,&sparm);
    avgloss+=l;
    if(l == 0) 
      correct++;
    else
      incorrect++;
    eval_prediction(i,testsample.examples[i],y,&model,&sparm,&teststats);

    if(empty_label(testsample.examples[i].y)) 
      { no_accuracy=1; } /* test data is not labeled */
    if(struct_verbosity>=2) {
      if((i+1) % 100 == 0) {
	printf("%ld..",i+1); fflush(stdout);
      }
    }
    free_label(y);
  }  
  avgloss/=testsample.n;
  fclose(predfl);

  if(struct_verbosity>=1) {
    printf("done\n");
    printf("Runtime (without IO) in cpu-seconds: %.2f\n",
	   (float)(runtime/100.0));    
  }
  if((!no_accuracy) && (struct_verbosity>=1)) {
    printf("Average loss on test set: %.4f\n",(float)avgloss);
    printf("Zero/one-error on test set: %.2f%% (%ld correct, %ld incorrect, %d total)\n",(float)100.0*incorrect/testsample.n,correct,incorrect,testsample.n);
  }
  print_struct_testing_stats(testsample,&model,&sparm,&teststats);
  free_struct_sample(testsample);
  free_struct_model(model);

  svm_struct_classify_api_exit();

  return(0);
}

void read_input_parameters(int argc,char *argv[],char *testfile,
			   char *modelfile,char *predictionsfile,
			   STRUCT_LEARN_PARM *struct_parm,
			   long *verbosity,long *struct_verbosity)
{
  long i;
  
  /* set default */
  strcpy (modelfile, "svm_model");
  strcpy (predictionsfile, "svm_predictions"); 
  (*verbosity)=0;/*verbosity for svm_light*/
  (*struct_verbosity)=1; /*verbosity for struct learning portion*/
  struct_parm->custom_argc=0;

  for(i=1;(i<argc) && ((argv[i])[0] == '-');i++) {
    switch ((argv[i])[1]) 
      { 
      case 'h': print_help(); exit(0);
      case '?': print_help(); exit(0);
      case '-': strcpy(struct_parm->custom_argv[struct_parm->custom_argc++],argv[i]);i++; strcpy(struct_parm->custom_argv[struct_parm->custom_argc++],argv[i]);break; 
      case 'v': i++; (*struct_verbosity)=atol(argv[i]); break;
      case 'y': i++; (*verbosity)=atol(argv[i]); break;
      default: printf("\nUnrecognized option %s!\n\n",argv[i]);
	       print_help();
	       exit(0);
      }
  }
  if((i+1)>=argc) {
    printf("\nNot enough input parameters!\n\n");
    print_help();
    exit(0);
  }
  strcpy (testfile, argv[i]);
  strcpy (modelfile, argv[i+1]);
  if((i+2)<argc) {
    strcpy (predictionsfile, argv[i+2]);
  }

  parse_struct_parameters_classify(struct_parm);
}

void print_help(void)
{
  printf("\nSVM-struct classification module: %s, %s, %s\n",INST_NAME,INST_VERSION,INST_VERSION_DATE);
  printf("   includes SVM-struct %s for learning complex outputs, %s\n",STRUCT_VERSION,STRUCT_VERSION_DATE);
  printf("   includes SVM-light %s quadratic optimizer, %s\n",VERSION,VERSION_DATE);
  copyright_notice();
  printf("   usage: svm_struct_classify [options] example_file model_file output_file\n\n");
  printf("options: -h         -> this help\n");
  printf("         -v [0..3]  -> verbosity level (default 2)\n\n");

  print_struct_help_classify();
}




