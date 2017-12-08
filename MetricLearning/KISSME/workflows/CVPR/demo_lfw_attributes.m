%% Face Verification on the Labeled Faces in the Wild (LFW) dataset [12] 
%
% [12] G. B. Huang, M. Ramesh, T. Berg, and E. Learned-Miller.
% Labeled faces in the wild: A database for studying face
% recognition in unconstrained environments. Technical Report
% 07-49, University of Massachusetts, Amherst, 2007
%
% See also http://vis-www.cs.umass.edu/lfw/
%
% Features:
%
% [13] N. Kumar, A. C. Berg, P. N. Belhumeur, and S. K. Nayar. 
% Attribute and Simile Classifiers for Face Verification. 
% In Proc. IEEE Intern. Conf. on Computer Vision, 2009.
%
% See also http://www.cs.columbia.edu/CAVE/databases/pubfig/

clc; clear all; close all;
DATA_OUT_DIR = fullfile('..','..','dataOut','cvpr','lfw','attributes');
run ../../toolbox/init;

load(fullfile(DATA_OUT_DIR,'lfw_attributes.mat'));

% algorithms for pairwise training
pair_metric_learn_algs = {...
    LearnAlgoKISSME(), ...
    LearnAlgoMahal(), ...
    LearnAlgoMLEuclidean(), ...
    %LearnAlgoITML(), ... % uncomment here to enable ITML
    %LearnAlgoLDML() ...  % uncomment here to enable LDML
    %LearnAlgoSVM(), ...  % uncomment here to enable SVM
    };

% algorithms to train with class labels
metric_learn_algs = { ...
    %LearnAlgoLMNN() ... % uncomment here to enable LMNN
    };

%% PREPROCESSING

params.pca.numCoeffs = 65; %we project onto the first 65 PCA dim. 
params.saveDir = fullfile(DATA_OUT_DIR,'out');
params.title ='LFW/ATTRIBUTES';
params.mu = 0; % smothing parameters see below
params.sigma =1;

%-- gaussian smoothing of the attribute features see [13] for details --%
X = attributes;
g = normpdf(X.*0.5,params.mu,params.sigma);
X = X .* g;

[ux,u,m] = applypca(X);

%% DO CROSS-VALIDATION ACCORDING TO LFW PROTOCOL

ds = CrossValidatePairs(struct(), pair_metric_learn_algs, pairs, ux(1:params.pca.numCoeffs,:), idxa, idxb);   
ds = CrossValidatePairs(ds,metric_learn_algs,pairs, ux(1:params.pca.numCoeffs,:), idxa, idxb, @pairsToLabels);

evalData(pairs, ds, params);
if isfield(params,'saveDir')
    save(fullfile(params.saveDir,'all_data.mat'),'ds','params');
end