%% Face Verification on the Public Figures (PubFig) Face Database [13]
%
% [13] N. Kumar, A. C. Berg, P. N. Belhumeur, and S. K. Nayar. 
% Attribute and Simile Classifiers for Face Verification. 
% In Proc. IEEE Intern. Conf. on Computer Vision, 2009.
%
% See also http://www.cs.columbia.edu/CAVE/databases/pubfig/
%
% Features:
%
% [13] N. Kumar, A. C. Berg, P. N. Belhumeur, and S. K. Nayar. 
% Attribute and Simile Classifiers for Face Verification. 
% In Proc. IEEE Intern. Conf. on Computer Vision, 2009.
%
% See also http://www.cs.columbia.edu/CAVE/databases/pubfig/

clc; clear all; close all;
run ../../toolbox/init;
DATA_OUT_DIR = fullfile('..','..','dataOut','cvpr','pubfig');

load(fullfile(DATA_OUT_DIR,'pubfig_attributes.mat'));

%% PREPROCESSING

params.pca.numCoeffs = 65;
params.title = 'PUBFIG/ATTRIBUTES';
params.mu = 0; % smothing parameters see below
params.sigma = 1;

%-- gaussian weighting see [13] for details --%
X = attributes;
g = normpdf(X.*0.5,params.mu,params.sigma);
X = X .* g;

%-- dimensionality reduction --%
[ux,u,m] = applypca(X);

%% LEARNING ALGORITHMS USED FOR CROSS VALIDATION

% algorithms for pairwise training
pair_metric_learn_algs = {...
    LearnAlgoKISSME(), ...
    LearnAlgoMahal(), ...
    LearnAlgoMLEuclidean(), ...
    %LearnAlgoSVM(), ...  % uncomment here to enable SVM
    %LearnAlgoITML(), ... % uncomment here to enable ITML
    %LearnAlgoLDML() ...  % uncomment here to enable LDML
    };

% algorithms to train with class labels
metric_learn_algs = { ...
    %LearnAlgoLMNN() ... % uncomment here to enable LMNN
    };

%% DO CROSS-VALIDATION ACCORDING TO LFW PROTOCOL

ds = CrossValidatePairs(struct(), pair_metric_learn_algs, pairs, ux(1:params.pca.numCoeffs,:), idxa, idxb);   
ds = CrossValidatePairs(ds,metric_learn_algs,pairs, ux(1:params.pca.numCoeffs,:), idxa, idxb, @pairsToLabels);

evalData(pairs, ds, params);
if isfield(params,'saveDir')
    save(fullfile(params.saveDir,'all_data.mat'),'ds','params');
end
