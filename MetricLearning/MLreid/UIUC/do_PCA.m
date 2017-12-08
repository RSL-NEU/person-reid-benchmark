function [pca_xdata] = do_PCA(xdata_new,partition,PCA_dim,k)
% Usage: This function learns PCA projection on the training data, and then
% apply to all data points.
% 
% Inputs: 
%   xdata_new: all the data points dxN, this function will load
%   previous stored training index to perform pca training
%
%   PCA_dim: the PCA dimension
%
%   k: the current trail to perform PCA
%
% Ouputs:
%   pca_xdata: pca_dim x N, all data points after apply pca
%
% Copyright (C) 2013 by Shiyu Chang and Zhen Li.

% load('IDX.mat','-mat');

% X_a = xdata_new(:,idx_cam_a{1});
% X_b = xdata_new(:,idx_cam_b{1});
%        
% X_a_train = X_a(:,idx_train(:,k));
% X_b_train = X_b(:,idx_train(:,k));
        
% xdata = [X_a_train X_b_train];

xdata = xdata_new(:,partition.idx_train);

mean_xdata = mean(xdata,2);
xdata_no_mean = xdata - repmat(mean_xdata,[1,size(xdata,2)]);

scater_train = xdata_no_mean'*xdata_no_mean;
scater_train = double(scater_train);

[e_vecter,~] =  eigs(scater_train,PCA_dim,'la');

mean_xdata_new = mean(xdata_new,2);
xdata_new_no_mean = xdata_new - repmat(mean_xdata_new,[1,size(xdata_new,2)]);

pca_xdata = (xdata_no_mean*e_vecter)'*xdata_new_no_mean;

end