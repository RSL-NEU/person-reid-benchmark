%% envrionment setup, need to be modifed accordingly

% modify the path point to the folder of datasets
datafolder = './Data';

% load vlfeat
run('./3rdParty/vlfeat-0.9.20/toolbox/vl_setup.m')

addpath(genpath('./FeatureExtraction'))
addpath(genpath('./MetricLearning'))
addpath(genpath('./util'))
addpath(genpath('./TrainTestSplits'))
addpath(genpath('./Evaluation'))

% make dir
mkdir('Results');