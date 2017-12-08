clc
clear 
%% Main file to run the experiment 

env_setup;

% parameters for data
params = { 'name','viper',...       % dataset name
           'datafolder',datafolder,...% folder for datasets
           'pair',[]};              % specific camera pairs
dopts = setParam('dataset',params);
% parameters for feature extraction
params = { 'featureType','whos',... % feature type
           'numRow',6,...           % number of split rows    
           'numCol',1,...           % number of split cols
           'overlap',0,...          % indicator for overlapping split (50%)
           'doPCA',0,...            % indicator for PCA dimension reduction
           'pcadim',100};           % PCA dimensions
fopts = setParam('feature',params);
% parameters for metric learning
params = { 'method','xqda',...      % metric learning method
           'kernels',[]};           % kernel types
mopts = setParam('metric',params);
% parameters for ranking
params = { 'rankType','rnp',...     % rank type for multi-shot
           'saveMetric',1,...       % indicator for saving learned metric
           'saveInterm',1};         % indicator for saving intermediate results
ropts = setParam('ranking',params);


% evaluate 
run_one_experiment;
           
           
           
