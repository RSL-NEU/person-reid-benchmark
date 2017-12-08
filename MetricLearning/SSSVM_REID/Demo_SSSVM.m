%% This is a demo of Sample-Sepcific SVM Learning for Person Re-identification.
%% If you find this code useful, please kindly cite the following paper.
%% Ying Zhang, Baohua Li, Huchuan Lu, Atsushi Irie, Xiang Ruan, 
%% Sample-Specific SVM Learning for Person Re-identification, CVPR2016.

close all; clear; clc;
addpath('functions');
%% dataset parameter
global par;
par = struct(...
    'dataset',      'VIPER', ... % 'VIPER','GRID'
    'numClass',      632,...    % VIPER-632, GRID-250
    'numAssit',      0,...    % VIPER-0, GRID-775
    'TRIAL',         10, ...    % average over 10 trials to obtain stable result     
    'numRanks',      100,...    % Number of Ranks
    'train_svm_c',   300, ...   % sample-specific svm  C  for positive set
    'wpos',          0.1, ...   % sample-specific wpos*C  for negative set
    'K',             316, ...   % dictionary size= Number of training samples 
    'nIter',         100, ...   % Iteration Number for LSSCDL
    'lambda1',       0.1,...    % lambda1*||Alphaw -Mx * Alphax||^2
    'lambdac',       0.01,...   % lambdac*||Alphaw||^2, lambdac*||Alphax||^2 
    'lambdam',       0.01,...   % lambdam*||Mx||^2
    'lambdad',       0.01,...   % lambdad*||Dx||^2, lambdad*||Dw||^2
    'epsilon',       5e-3);     % convergence 

%% Probe and Gallery Index
% indp=1:2:(par.numClass*2);
% indg=2:2:(par.numClass*2);
% indp = 1:632;
% indg = 633:par.numClass*2;
%% Data loading
load('feature_viper_LOMO_6patch.mat')
load('Partition_viper.mat')
% features = features';
% galFea = features(:,indg);
% probFea = features(:,indp);
% feaFile =['./data/',par.dataset,'_lomo.mat'];
% load(feaFile, 'FeatSetC');
% galFea = FeatSetC(:,indg);
% probFea =FeatSetC(:,indp);

%% Random seed and evaluation
seed = 0;
rng(seed);
cms = zeros(par.TRIAL, par.numRanks);

%% Average Results for 10 trials
for trial = 1 : par.TRIAL
    train = features(partition(trial).idx_train,:);
    ID_train = personID(partition(trial).idx_train);
    cam_train = camID(partition(trial).idx_train);
    
    test = features(partition(trial).idx_test,:);
    ID_test = personID(partition(trial).idx_test);
    cam_test = camID(partition(trial).idx_test);
    
    X_a_train = train(cam_train==1,:)';
    X_b_train = train(cam_train==2,:)';
    label = ID_train(cam_train==1);
    
    X_a_test = test(cam_test==1,:)';
    X_b_test = test(cam_test==2,:)';
    
%     p = randperm(par.numClass);
%     X_a_train = probFea(:,p(1:par.numClass/2));
%     X_b_train = galFea(:,p(1:par.numClass/2));
%     X_a_test = probFea(:,p(par.numClass/2+1: end));
%     X_b_test =galFea(:,p(par.numClass/2+1: end));
%     if par.numAssit>0
%     X_b_test = [X_b_test FeatSetC(:,par.numClass*2+1:end)];
%     end
%     label = (1:size(X_a_test,2))';   
    %% Dimensionality reduction
    P=LDA(X_b_train', X_a_train', label, label); 
    X_a_train=P*X_a_train;
    X_b_train=P*X_b_train;
    X_a_test=P*X_a_test;
    X_b_test=P*X_b_test;
    %% Train    
    par.K=size(X_a_train,2);
    [W, Response]=SpecificSVMLearn(X_a_train, X_b_train,label,label,par);
    Dict=JointDictLearning(X_a_train, W,par);  
    %% Test
    fprintf('Testing...');
    dist=MatchScore(X_a_test,X_b_test,Dict,par); 
    fprintf('Done!\n');
   %% Evaluation 
    cms(trial,:) = EvalCMC( -dist, 1:size(X_b_test,2), 1:size(X_a_test,2), par.numRanks);   
    fprintf('Fold %d: Rank1,  Rank5, Rank10, Rank15, Rank20\n',trial);
    fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', cms(trial,[1,5,10,15,20]) * 100);
end

meanCms = mean(cms);
plot(1 : par.numRanks, meanCms);
axis([1 20 0 1]);
grid on

fprintf('The average performance:\n');
fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', meanCms([1,5,10,15,20]) * 100);
