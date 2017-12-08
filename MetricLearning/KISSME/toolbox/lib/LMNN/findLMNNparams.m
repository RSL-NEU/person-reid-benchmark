function [K,mu,outdim,maxiter]=findLMNNparams(xTr,yTr,knn,varargin)
%function [K,mu,outdim,maxiter]=findLMNNparams(xTr,yTr,knn,varargin)
% This function automatically finds the best hyper-parameters for LMNN
% Please see demo2.m for a use case. 
%
% copyright Kilian Weinberger 2015

addpath(genpath('autoLMNN'));
setpaths;
startup;

%% create valiadation data set 
[train,val]=makesplits(yTr,0.8,1,1);
%% Setting parameters for Bayesian Global Optimization
opt = defaultopt(); % Get some default values for non problem-specific options.
opt.dims = 4; % Number of parameters.
%%min/max for K  MU      OUTDIM  MAXITER
opt.mins =  [ 1,  0           5      50]; % Minimum value for each of the parameters. Should be 1-by-opt.dims
opt.maxes = [ 5,  1 size(xTr,1)    2000]; % Vector of maximum values for each parameter. 
opt.max_iters = 12; % How many parameter settings do you want to try?
opt.grid_size = 20000;
opt=extractpars(varargin,opt);

%% Start the optimization
F = @(P) optimizeLMNN(xTr(:,train),yTr(train),xTr(:,val),yTr(val),knn,P); % CBO needs a function handle whose sole parameter is a vector of the parameters to optimize over.
[bestP,mv,T] = bayesopt(F,opt);   % ms - Best parameter setting found
                               % mv - best function value for that setting L(ms)
                               % T  - Trace of all settings tried, their function values, and constraint values.


K=round(bestP(1));
mu=bestP(2);
outdim=ceil(bestP(3));
maxiter=ceil(bestP(4));
fprintf('\nBest parameters: K=%i mu=%2.4f outdim=%i maxiter=%i!\n',K,mu,outdim,maxiter);


function valerr=optimizeLMNN(xTr,yTr,xVa,yVa,knn,P)
% function valerr=optimizeLMNN(xTr,yTr,xVa,yVa,P);

mu=P(2);
outdim=ceil(P(3));
K=round(P(1));
maxiter=ceil(P(4));
fprintf('\nTrying K=%i mu=%2.4f outdim=%i maxiter=%i ... ',K,mu,outdim,maxiter);
[L,~] = lmnn2(xTr, yTr,K,'maxiter',maxiter,'quiet',1,'outdim',outdim,'mu',mu,'subsample',1.0);
valerr=knncl(L,xTr,yTr,xVa,yVa,knn,'train',0);
fprintf('validation error=%2.4f\n',valerr);
