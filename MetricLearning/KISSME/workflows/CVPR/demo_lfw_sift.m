%% Face Verification on the Labeled Faces in the Wild (LFW) dataset [12]
%
% Dataset:
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
% [9] M. Guillaumin, J. Verbeek, and C. Schmid. Multiple instance
% metric learning from automatically labeled bags of faces. In
% Proc. European Conf. on Computer Vision, 2010.

clc; clear all; close all;
DATA_OUT_DIR = fullfile('..','..','dataOut','cvpr','lfw','sift');
run ../../toolbox/init;

load(fullfile(DATA_OUT_DIR,'lfw_sifts.mat'));

%% PARAMS

params.numCoeffs = 100;
params.saveDir = fullfile(DATA_OUT_DIR,'all');
params.title = 'LFW/SIFT';

% LEARNING ALGORITHMS USED FOR CROSS VALIDATION

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
    %LearnAlgoLMNN() ...  % uncomment here to enable LMNN
    };

%% DO CROSS-VALIDATION ACCORDING TO LFW PROTOCOL (LEAVE-ONE-OUT)

ds = CrossValidatePairs(struct(), pair_metric_learn_algs, pairs, ux(1:params.numCoeffs,:), idxa, idxb);   
ds = CrossValidatePairs(ds,metric_learn_algs,pairs, ux(1:params.numCoeffs,:), idxa, idxb, @pairsToLabels);

evalData(pairs, ds, params);
if isfield(params,'saveDir')
    save(fullfile(params.saveDir,'all_data.mat'),'ds','params');
end