%% svm_light .o files
fprintf('doing hideo \n');
mex -largeArrayDims  -c  -DWIN ./svm_light/svm_hideo.c
fprintf('doing learn \n');
mex -largeArrayDims  -c  -DWIN ./svm_light/svm_learn.c
fprintf('doing common \n');
mex -largeArrayDims  -c  -DWIN ./svm_light/svm_common.c

%% svm_struct .o files
mex -largeArrayDims  -c -DWIN ./svm_struct/svm_struct_learn.c
mex -largeArrayDims  -c -DWIN ./svm_struct/svm_struct_common.c

%% svm_struct - custom  .o files
mex -largeArrayDims  -c -DWIN ./svm_struct_api.c 
mex -largeArrayDims  -c -DWIN ./svm_struct_learn_custom.c

mex -largeArrayDims -DWIN -output  svm_struct_learn svm_struct_learn_mex.c svm_struct_api.obj  svm_struct_learn_custom.obj svm_struct_learn.obj svm_struct_common.obj svm_common.obj svm_learn.obj svm_hideo.obj 

% delete *.obj
