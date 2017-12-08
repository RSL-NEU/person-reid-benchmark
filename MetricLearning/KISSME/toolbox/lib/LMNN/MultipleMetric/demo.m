echo off;
clear all;
clc;
rand('seed',1);
setpaths
fprintf('Loading data ...\n');
load('data/digits.mat');       

% To speed up the demo, I run lmnn with k=1. 
% Sometimes k=3 is slightly better. 
%
                 
fprintf('Running single metric LMNN (with dimensionality reduction from 50d to 15d)  ...\n');
[Ldim,Det]=lmnn24(xTr,yTr,1,'outdim',15,'quiet',1,'maxiter',500,'validation',0.3,'checkup',0);
enerrdim=energyclassify(Ldim,xTr,yTr,xTe,yTe,3);
knnerrLdim=knnclassifytreeomp(Ldim,xTr,yTr,xTe,yTe,3);
knnerrI=knnclassifytreeomp(eye(size(xTr,1)),xTr,yTr,xTe,yTe,3);

fprintf('3-NN Euclidean training error: %2.2f\n',knnerrI(1)*100);
fprintf('3-NN Euclidean testing error: %2.2f\n',knnerrI(2)*100);
fprintf('15-dim usps digits data set (after dim-reduction):\n');
fprintf('3-NN Malhalanobis training error: %2.2f\n',knnerrLdim(1)*100);
fprintf('3-NN Malhalanobis testing error: %2.2f\n',knnerrLdim(2)*100);
fprintf('\nEnergy classification error: %2.2f\n',enerrdim*100);
fprintf('\nTraining time: %2.2fs\n\n\n',Det.time);


fprintf('Running single metric LMNN  ...\n');
[L,Det]=lmnn24(xTr,yTr,1,'quiet',1,'maxiter',500,'validation',0.3,'checkup',0);
enerr=energyclassify(L,xTr,yTr,xTe,yTe,3);
knnerrL=knnclassifytreeomp(L,xTr,yTr,xTe,yTe,3);


clc;
fprintf('100-dim usps digits data set:\n');
fprintf('3-NN classification:\n')
fprintf('Training:\tEuclidean=%2.2f\t1-Metric(dim-red)=%2.2f\t1-Metric(square)=%2.2f\n',knnerrI(1)*100,knnerrLdim(1)*100,knnerrL(1)*100)
fprintf('Testing:\tEuclidean=%2.2f\t1-Metric(dim-red)=%2.2f\t1-Metric(square)=%2.2f\n',knnerrI(2)*100,knnerrLdim(2)*100,knnerrL(2)*100)
fprintf('\nEnergy classification:\n')
fprintf('Testing:\t1-Metric(dim-red)=%2.2f\t1-Metric(square)=%2.2f\n',enerrdim*100,enerr*100);
fprintf('\nTraining time: %2.2fs\n\n',Det.time);


%% GB-LMNN
fprintf('\n\nRunning GB-LMNN  ...\n');
% to speed up the demo I set "validation" to 0.0 -- this only works because maxiter is so low
% usually you would want to hvae maxiter larger, and 'validation' set to 0.2 or 0.3. 
embed=gb_lmnn2(xTr,yTr,3,Ldim);
fprintf('Using gb-lmnn for classification ...\n')
gberr=knncl([],embed(xTr), yTr,embed(xTe),yTe,3);
%%
clc;
fprintf('50-dim usps digits data set:\n');
fprintf('3-NN classification:\n')
fprintf('Training:\tEuclidean=%2.2f\t1-Metric(dim-red)=%2.2f\t1-Metric(full)=%2.2f\tGB-LMNN=%2.2f\n',knnerrI(1)*100,knnerrLdim(1)*100,knnerrL(1)*100,gberr(1)*100)
fprintf('Testing:\tEuclidean=%2.2f\t1-Metric(dim-red)=%2.2f\t1-Metric(full)=%2.2f\tGB-LMNN=%2.2f\n',knnerrI(2)*100,knnerrLdim(2)*100,knnerrL(2)*100,gberr(2)*100)
fprintf('\nEnergy classification:\n')
fprintf('Testing:\t1-Metric(dim-red)=%2.2f\t1-Metric(full)=%2.2f\n',enerrdim*100,enerr*100);
%fprintf('\nTraining time: %2.2fs (My Mac Mini requires 127s)\n\n',Dets.time+Det.time);


       
