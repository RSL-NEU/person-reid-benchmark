/***********************************************************************/
/*                                                                     */
/*   svm_struct_learn_custom.c (instantiated for SVM-perform)          */
/*                                                                     */
/*   Allows implementing a custom/alternate algorithm for solving      */
/*   the structual SVM optimization problem. The algorithm can use     */ 
/*   full access to the SVM-struct API and to SVM-light.               */
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 09.01.08                                                    */
/*                                                                     */
/*   Copyright (c) 2008  Thorsten Joachims - All rights reserved       */
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


void svm_learn_struct_joint_custom(SAMPLE sample, STRUCT_LEARN_PARM *sparm,
				   LEARN_PARM *lparm, KERNEL_PARM *kparm, 
				   STRUCTMODEL *sm)
     /* Input: sample (training examples)
	       sparm (structural learning parameters)
               lparm (svm learning parameters)
               kparm (kernel parameters)
	       Output: sm (learned model) */
{
  /* Put your algorithm here. See svm_struct_learn.c for an example of
     how to access this API. */
}

