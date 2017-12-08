 function [ feature_vec ] = GOG( X, param )
% Gaussian of Gaussian (GOG) descriptor
% 
% Input: 
%  <X>: input RGB image. Size: [h, w, 3]
%  [param]: parameters.
%       lfparam: paramters for pixel feature
%       p: intervals of patch extraction. Default: 2
%       k: size of patch (k x k pixles). Default: 5
%       epsilon0: regularization paramter of covariance. Default: 0.001
%       ifweight: patch weight  0 -- not use,  1 -- use. Default: 1 
%       G: number of horizontal strips. Default: 7
% 
% Output:
%  [feature_vec]: the extracted GOG descriptor. Size:[parGrid.numgrid*param.dimension, 1]
%
% Usage:
%  addpath('GOG/mex');
%  f = 1; % 1 -- GOG_RGB, 2 -- GOG_Lab, 3 -- GOG_HSV, 4 -- GOG_nRnG
%  param = set_default_parameter(f); 
%  I = imread('X.bmp'); % load image 
%  feature_vec = GOG(I, param); 
%
% Version: 1.0
% Date: 2016-5-16
%
% Reference: 
%   T. Matsukawa, T. Okabe, E. Suzuki, Y. Sato, 
%   "Hierarchical Gaussian Descriptor for Person Re-Identification", 
%   In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp.1363--1372, 2016.
%
% Author: Tetsu Matsukawa
% Institute: Faculty of Information Science and Electrical Engineering, Kyushu University, Japan. 
% Email: matsukawa@inf.kyushu-u.ac.jp

if isfield(param, 'patchSize')
    % pre-defined patch size and step size
    parGrid.gwidth = param.patchSize(2);
    parGrid.gheight = param.patchSize(1);
    parGrid.xstep = param.gridSize(2);
    parGrid.ystep = param.gridSize(1);
else    
    if param.G == 7
        % 7x1 horizontal strips
        parGrid.gwidth = size(X,2);
        parGrid.gheight = size(X,1)/4;
        parGrid.ystep = parGrid.gheight/2;
        parGrid.xstep = parGrid.gwidth;
    elseif param.G == 6
        % 6x1 non-overlap horizontal strips
        parGrid.gwidth = size(X,2);
        parGrid.gheight = floor(size(X,1)/6);
        parGrid.ystep = parGrid.gheight;
        parGrid.xstep = parGrid.gwidth;
    elseif param.G == 15
        % 15x1 horizontal strips
        parGrid.gwidth = size(X,2);
        parGrid.gheight = 16;
        parGrid.ystep = 8;
        parGrid.xstep = parGrid.gwidth;
    elseif param.G == 29
        % 15x1 horizontal strips
        parGrid.gwidth = size(X,2);
        parGrid.gheight = 16;
        parGrid.ystep = 4;
        parGrid.xstep = parGrid.gwidth;
    else    
        fprintf('This region configuration is not defined \n');
    end
end
%% 1. Pixel Feature Extraction
F = get_pixelfeatures(X, param.lfparam );

%% 2. Patch Gaussians
FCorr = create_corr_MT(F); % upper triangle part of F*F^T for each pixel ( Size[ h, w, d*(d+1)/2 ] )
% intergral histgram (IH)
IH_Corr = create_IH_MT(FCorr);
IH_F = create_IH_MT(F);

halffsize = (param.k - 1)/2;
[cols, rows] = meshgrid(1:param.p:size(F,2), 1:param.p:size(F,1));
cols = cols(:);
rows = rows(:);

xlefts = max( cols + 1 - halffsize, 2);
xrights = min( cols + 1 + halffsize, size(IH_Corr, 2));
yups = max( rows + 1 - halffsize, 2);
ydowns = min( rows + 1  + halffsize, size(IH_Corr, 1));

IH_Corr2 = reshape(IH_Corr, size(IH_Corr, 1)*size(IH_Corr, 2), size(IH_Corr, 3));
IH_F2 = reshape(IH_F, size(IH_F, 1)*size(IH_F, 2), size(IH_F, 3));

points1 = (xrights -1)*size(IH_Corr,1) + ydowns;
points2 = (xlefts-2) *size(IH_Corr, 1) + yups-1;
points3 = (xrights-1)*size(IH_Corr, 1) + yups-1;
points4 = (xlefts-2)*size(IH_Corr, 1) + ydowns;

sumFCorrs = IH_Corr2(points1, :) + IH_Corr2(points2, :) - IH_Corr2(points3, :) - IH_Corr2(points4, :);  % get values from IH
sumFs = IH_F2(points1, :) + IH_F2(points2, :) - IH_F2(points3, :) - IH_F2(points4, :); % get values from IH
sumpixels = (xrights - xlefts + 1).*(ydowns -yups + 1);

Pcells = cell( numel(cols), 1);
for i = 1:numel(cols)
    sumFCorr = sumFCorrs(i,:)';
    sumF = sumFs(i,:)';
    sumpixel = sumpixels(i);
    
    
    S = ( vec2mat( sumFCorr, param.d ) - sumF*sumF'./sumpixel)./(sumpixel -1 ); % covariance matrix by integral image 
    S = S + param.epsilon0.*max(trace(S), 0.01).*eye(size(S)); % regularizaiton 

    
    meanVec = sumF./sumpixel; % mean vector 
    Pcells{i} = power( det(S),  -1/(size(S, 1) + 1)).*[ S+meanVec*meanVec' meanVec; meanVec'  1]; % patch Gaussian matrix
end

logPcells = cell2mat(cellfun( @(x) halfvec(logm(x))', Pcells, 'un', 0)); % apply log-Euclidean and half-vectorization 


F2 = reshape(logPcells, size(F,1)/param.p, size(F,2)/param.p, size(logPcells, 2) );

%% 3.  Region Gaussians 
% setup of patch weight
H0 = size(F2, 1); W0 = size(F2, 2);

if param.ifweight == 0
    weightmap = ones(H0, W0);
end
if param.ifweight == 1
    weightmap = zeros( H0, W0);
    sigma = double(W0)/4;
    mu = double(W0)/2;
    for x=1:W0
        weightmap(:, x) = exp( -double((x-mu)*(x-mu)/(2*sigma*sigma)))/(sigma*sqrt(2*pi));
    end
end

gheight2 = parGrid.gheight/param.p; gwidth2 = parGrid.gwidth/param.p;
ystep2 = parGrid.ystep/param.p; xstep2 = parGrid.xstep/param.p;

[cols, rows] = meshgrid(1:xstep2:size(F2,2)-gwidth2+1, 1:ystep2:size(F2,1)-gheight2+1);
cols = cols(:);
rows = rows(:);

xlefts = cols;
xrights = cols + gwidth2 -1;
yups = rows;
ydowns = rows + gheight2 - 1;

Pcells = cell( numel(cols), 1);
for i = 1:numel(cols)
    weightlocal = weightmap( yups(i):ydowns(i), xlefts(i):xrights(i), :);
    F2localorg =  F2(yups(i):ydowns(i), xlefts(i):xrights(i), :);
    
    weightlocal = reshape(weightlocal, size(weightlocal, 1)*size(weightlocal, 2), 1);
    F2local = reshape( F2localorg, size(F2localorg, 1)*size(F2localorg, 2), size(F2localorg, 3));
    
    wF2local = repmat(weightlocal, 1, size(F2local, 2)).*F2local;
    meanVec = sum(wF2local)'./sum(weightlocal); % mean vector
    
    tmp = F2local - repmat(meanVec', size(F2local, 1), 1);
    FCorr2 = create_corr_MT( reshape(tmp, size(F2localorg, 1), size(F2localorg, 2), size(tmp, 2)));
    FCorr2 = reshape(FCorr2, size(FCorr2, 1)*size(FCorr2, 2), size(FCorr2, 3));
    
    wFCorr2 = repmat(weightlocal, 1, size(FCorr2, 2)).*FCorr2;
    fcov =  sum(wFCorr2)./sum(weightlocal);
    S = vec2mat(fcov', size(F2, 3) ); % covariance matrix 
    
    S = S + param.epsilon0.*trace(S).*eye(size(S) ); % regularizaiton 
    Pcells{i} = power( det(S), - 1/(size(S, 1) + 1)).*[ S+meanVec*meanVec' meanVec; meanVec'  1]; % apply region Gaussian matrix
end

logPcells = cell2mat( cellfun( @(x) halfvec(logm(x))', Pcells, 'un', 0) ); % apply log Euclidean and half-vectorization
feature_vec = reshape(logPcells, size(logPcells, 1)*size(logPcells, 2), 1 ); % concatenate feature vector of each grid

end

