%% LMNN Demo
%
% This demo tests various flavors of LMNN
%% Initialize path anc clear screen
clear
close all
install;
setpaths;
figure;
clc;
rand('seed',1);

%% Load data 
load data/segment.mat;

%% KNN classification error before metric learning  
errRAW=knncl([],xTr, yTr,xTe,yTe,1);fprintf('\n');
fprintf('\n')

%% KNN classification after PCA
L0=pca(xTr)';
errPCA=knncl(L0(1:3,:),xTr, yTr,xTe,yTe,1);fprintf('\n');
subplot(3,2,1);
scat(L0*xTr,3,yTr);
title(['PCA Training (Error: ' num2str(100*errPCA(1),3) '%)'])
noticks;box on;
subplot(3,2,2);
scat(L0*xTe,3,yTe);
title(['PCA Test (Error: ' num2str(100*errPCA(2),3) '%)'])
noticks;box on;
drawnow


% Call LMNN to get the initiate linear transformation
fprintf('\n')
disp('Learning initial metric with LMNN ...')
[L,~] = lmnn3(xTr, yTr,3,L0,'maxiter',1000,'quiet',1,'outdim',3,'mu',0.5,'validation',0.2,'earlystopping',25,'subsample',0.3);
% KNN classification error after metric learning using LMNN
errL=knncl(L,xTr, yTr,xTe,yTe,1);fprintf('\n');

% Plotting LMNN embedding
subplot(3,2,3);
scat(L*xTr,3,yTr);
title(['LMNN Training (Error: ' num2str(100*errL(1),3) '%)'])
noticks;box on;
drawnow
subplot(3,2,4);
scat(L*xTe,3,yTe);
title(['LMNN Test (Error: ' num2str(100*errL(2),3) '%)'])
noticks;box on;
drawnow


% Gradient boosting
fprintf('\n')
disp('Learning nonlinear metric with GB-LMNN ... ')
embed=gb_lmnn(xTr,yTr,3,L,'ntrees',200,'verbose',false,'XVAL',xVa,'YVAL',yVa);

% KNN classification error after metric learning using gbLMNN
errGL=knncl([],embed(xTr), yTr,embed(xTe),yTe,1);fprintf('\n');
subplot(3,2,5);
scat(embed(xTr),3,yTr);
title(['GB-LMNN Training (Error: ' num2str(100*errGL(1),3) '%)'])
noticks;box on;
drawnow
subplot(3,2,6);
scat(embed(xTe),3,yTe);
title(['GB-LMNN Test (Error: ' num2str(100*errGL(2),3) '%)'])
noticks;box on;
drawnow
disp('Dimensionality Reduction Demo:')
disp(['1-NN Error for raw (high dimensional) input is : ',num2str(100*errRAW(2),3),'%']);
disp(['1-NN Error after PCA in 3d is : ',num2str(100*errPCA(2),3),'%']);
disp(['1-NN Error after LMNN in 3d is : ',num2str(100*errL(2),3),'%']);
disp(['1-NN Error after gbLMNN in 3d is : ',num2str(100*errGL(2),3),'%']);

