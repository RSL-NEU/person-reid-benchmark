function [ param ] = set_default_parameter( lf_type )
% set default parameter 
% input:
%        lf_type (0--yMthetaRGB, 1--yMthetaLab, 2--yMthetaHSV, 3--yMthetanRnG)
% Output: 
%         param  paramters for GOG. 
param = struct;
param.epsilon0 = 0.001; % reguralization paramter of covariance
param.p = 2; % patch extraction interval
param.k = 5; % patch size (k x k pixels) 
param.ifweight = 1; % patch weight  0 -- not use,  1 -- use.  
param.G = 7; % number of horizontal strips

[ param.lfparam.num_element, param.lfparam.lf_name, param.lfparam.usebase ] = set_pixelfeatures( lf_type );
param.d = param.lfparam.num_element; % dimension of pixel features
param.m = (param.d*param.d + 3*param.d )/2 + 1; % dimension of patch Gaussian vector
param.dimlocal = (param.m*param.m + 3*param.m)/2 + 1; % dimension of region Gaussian vector
param.dimension = param.G*param.dimlocal; % dimension of feature vector

param.name = strcat('GOG', param.lfparam.lf_name);
end

