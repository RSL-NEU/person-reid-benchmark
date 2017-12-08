function [Method] = svmml_learn_full_final(X, Y, AlgoOption)
% USAGE: [A, b] = svmml_learn(X, Y, p, lambda1,lambda2, loss, maxit, verbose, initAb)
% SVM metric learning.
%
% Input:
%        X --- N x d matrix, n data points in a d-dim space
%        Y --- N x 1 matrix, labels of n data points
%        p --- if specified (not empty), learn a d x p projection matrix;
%              otherwise learn a d x d full matrix
%        lambda1 --- a positive coefficient for regulerization on A
%        lambda2 --- a positve coefficient for regulerization on B
%        loss   --- loss function type (default=1)
%                   1 - L2 regularized, hinge loss
%                   2 - L2 regularized, logistic regression
%        maxit   --- maximum number of iterations until stop (default=30)
%        verbose --- whether to verbosely display the learning process
%                    (default=false)
%        initAb  --- initial value for [A(:); b] (default=random)
%
% Output:
%        A --- d x d or d x p matrix
%        b --- the estimated bias term
%
% Copyright (C) 2013 by Shiyu Chang.
% Modified by Mengran Gou @ 2014, if you use this code, please cite the
% following paper:
% Li, Z., Chang, S., Liang, F., Huang, T.S., Cao, L., Smith, J.R.: Learning 
% locally-adaptive decision functions for person verification. In CVPR 2013
p = AlgoOption.p;
lambda1 = AlgoOption.lambda1;
lambda2 = AlgoOption.lambda2;
maxit = AlgoOption.maxit;
verbose = AlgoOption.verbose;
initMNb = [];

% Method = struct('rbf_sigma',0);
% [K, Method] = ComputeKernel(X, AlgoOption.kernel, Method);
K = double(X);

[n,d] = size(K);

assert(numel(Y) == n);

if isempty(p)
    p = d;
end

assert(p > 0 && p <= d)

if ~exist('initMNb','var') || isempty(initMNb)
    initM = rand(d,p,class(K))/d;
    initN = rand(d,p,class(K))/d;
    
    % since we never learn the full martix of A
    initM = orth(double(initM));
    initN = orth(double(initN));
    assert(size(initM,2)==p);
    assert(size(initN,2)==p);
    
    % cascade initM, initN, and b together
    initMNb = [initM(:); initN(:); rand/d];
else
    assert(isequal(size(initMNb),[d*p*2+1,1]));
end


loss_func = 'svmml_full_gradient_final';

% convert M,N to A,B 
M = reshape(initMNb(1:d*p),d,p);
N = reshape(initMNb(d*p+1:end-1),d,p);
b = initMNb(end);
A = M*M';
B = N*N';
initABb = [A(:); B(:); b];

%% Start optimization

xstarbest = minimize(initABb, loss_func, maxit, K, Y, lambda1,lambda2);

A = reshape(xstarbest(1:d^2),d,d);
B = reshape(xstarbest(d^2+1:end-1),d,d);
b = xstarbest(end);

% ensure symmetry for A and B
A = (A + A')/2;
B = (B + B')/2;

Method.name = 'svmml';
Method.A = A;
Method.B = B;
Method.b = b;
Method.Prob = [];
Method.Ranking = [];
Method.Dist = [];
Method.Trainoption = AlgoOption;
end





