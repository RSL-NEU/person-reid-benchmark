%% Person Re-Identification on the VIPeR dataset [6]
%
% [6] D. Gray, S. Brennan, and H. Tao. Evaluating appearance
% models for recongnition, reacquisition and tracking. In Proc.
% IEEE Intern.Workshop on Performance Evaluation of Tracking
% and Surveillance, 2007.
%
% See also http://vision.soe.ucsc.edu/?q=node/178
%
% Features:
%
% HSV, Lab histograms and LBPs [16] to describe color and texture of the
% overlapping blocks (size 8x16 and stride of 8x8 pixels). The image 
% descriptor is a concatenation of the local ones. Using PCA the descriptor
% is projected onto a 34 dimensional subspace.
% 

%clc; clear all; close all;
DATA_OUT_DIR = fullfile('..','..','dataOut','cvpr','viper');
run('../../toolbox/init.m');

%% Set up parameters

params.numCoeffs = 500; %dimensionality reduction by PCA to 34 dimension
params.N = 632; %number of image pairs, 316 to train 316 to test
params.numFolds = 10; %number of random train/test splits
params.saveDir = fullfile(DATA_OUT_DIR,'all');
params.pmetric = 0;

%% Load Features

load(fullfile(DATA_OUT_DIR,'viper_features.mat'));

%% Cross-validate over a number of runs

pair_metric_learn_algs = {...
    LearnAlgoKISSME(params), ...
    LearnAlgoMahal(), ...
    LearnAlgoMLEuclidean(), ...
    LearnAlgoITML(), ... 
    LearnAlgoLDML(), ... 
    LearnAlgoLMNN() ...  
    };

[ ds ] = CrossValidateViper(struct(), pair_metric_learn_algs,ux(1:params.numCoeffs,:),idxa,idxb,params);

%% Plot Cumulative Matching Characteristic (CMC) Curves

names = fieldnames(ds);
for nameCounter=1:length(names)
   s = [ds.(names{nameCounter})];
   ms.(names{nameCounter}).cmc = cat(1,s.cmc)./(params.N/2);
   ms.(names{nameCounter}).roccolor = s(1).roccolor;
end

h = figure;
names = fieldnames(ms);
for nameCounter=1:length(names)
   hold on; plot(median(ms.(names{nameCounter}).cmc,1),'LineWidth',2, ...
       'Color',ms.(names{nameCounter}).roccolor);
end
  
title('Cumulative Matching Characteristic (CMC) Curves - VIPeR dataset');
box('on');
set(gca,'XTick',[0 10 20 30 40 50 100 150 200 250 300 350]);
ylabel('Matches');
xlabel('Rank');
ylim([0 1]);
hold off;
grid on;
legend(upper(names),'Location','SouthEast');

if isfield(params,'saveDir')
    exportAndCropFigure(h,'all_viper',params.saveDir);
    save(fullfile(params.saveDir,'all_data.mat'),'ds');
end
