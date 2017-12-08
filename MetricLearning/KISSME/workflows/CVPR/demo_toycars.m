%% Comparing never seen objects on the ToyCars dataset [15]
%
% [15] E. Nowak and F. Jurie. Learning visual similarity measures
% for comparing never seen objects. In Proc. IEEE Intern.
% Conf. on Computer Vision and Pattern Recognition, 2007.
%
% See also http://lear.inrialpes.fr/people/nowak/similarity/index.html
%
% Features:
%
% HSV, Lab histograms and LBPs [16] to describe color and texture of the
% non-overlapping blocks (size 30x30). The global image descriptor is a 
% concatenation of the local ones. Using PCA the descriptor
% is projected onto a 50 dimensional subspace.
%

%-- INIT / LOAD --%
clc; clear all; close all;
run ../../toolbox/init;
DATA_OUT_DIR = fullfile('..','..','dataOut','cvpr','toy_car_lear');

load(fullfile(DATA_OUT_DIR,'toycars_features.mat'));

%-- PARAMS --%

params.pca.numDims = 50; %we project onto the first 50 PCA dim. 
params.svm.liblinear_options = '-B 10 -s 2 -c 1';
params.svm.smoothing = 0;
params.saveDir = fullfile(DATA_OUT_DIR,'out');

%% VALIDATE AND PLOT ROC CURVES

% algorithms for pairwise training
pair_metric_learn_algs = {...
    LearnAlgoKISSME(), ...
    LearnAlgoMahal(), ...
    LearnAlgoMLEuclidean(), ...
    LearnAlgoSVM(params.svm), ...  
    LearnAlgoITML(), ...           
    LearnAlgoLDML() ...                
    };

% algorithms to train with class labels
metric_learn_algs = { ...
    LearnAlgoLMNN() ... 
    };

%% DO VALIDATION ACCORDING TO TOYCARS PROTOCOL
ds = CrossValidatePairs(struct(),pair_metric_learn_algs, pairs, ux(1:params.pca.numDims,:), idxa, idxb);   
ds = CrossValidatePairs(ds,metric_learn_algs,pairs, ux(1:params.pca.numDims,:), idxa, idxb, @ToyCarPairsToLabels); 

%% EVALUATION
% we evaluate only on the test set (fold 2), train set (fold 1).
[ignore, rocPlot] = evalData(pairs(~logical([pairs.training])), ds(2), params);
hold on; plot(1-0.859,0.859,'+','Color',[0.5 0.5 1],'LineWidth',2);
legendEntries = get(rocPlot.hL,'String');
legendEntries{end+1} = 'Nowak (0.859)';
legend(gca(rocPlot.h),legendEntries,'Location', 'SouthEast');
title('ROC Curves ToyCars');

if isfield(params,'saveDir')
    exportAndCropFigure(rocPlot.h,'all_toycars',params.saveDir);
    save(fullfile(params.saveDir,'all_data.mat'),'ds');
end