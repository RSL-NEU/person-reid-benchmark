function [M,N,b] = svmml_learn_proj_final(X, Y, p, lambda1,lambda2, maxit, ...
    verbose,initMNb)
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

[n,d] = size(X);

assert(numel(Y) == n);


assert(p > 0 && p <= d)

if ~exist('initMNb','var') || isempty(initMNb)
    initM = rand(d,p,class(X))/d;
    initN = rand(d,p,class(X))/d;
    initM = orth(double(initM));
    initN = orth(double(initN));
    assert(size(initM,2)==p);
    assert(size(initN,2)==p);
    
    % cascade initM, initN, and b together
    initMNb = [initM(:); initN(:); rand/d];
else
    assert(isequal(size(initMNb),[d*p*2+1,1]));
end


loss_func = 'svmml_projection_gradient_final';


%% Start optimization

xstarbest = minimize(initMNb, loss_func, maxit, verbose, X, Y, p,lambda1,lambda2);

M = reshape(xstarbest(1:d*p),d,p);
N = reshape(xstarbest(d*p+1:end-1),d,p);
b = xstarbest(end);




end





