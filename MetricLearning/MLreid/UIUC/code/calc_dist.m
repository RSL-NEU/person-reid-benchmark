function D = calc_dist(X, A)
% Calculate pairwise Mahalanobis distance or scaled Euclidean distance:
%    D_ij = (x_i - x_j)' * A * (x_i - x_j)
%
% Input:
%        X --- N x d matrix, n data points in a d-dim space
%        A --- d x d matrix for Mahalanobis distance
%           or scaler for scaled Euclidean distance
%
% Output:
%        D --- N x N distance matrix
%
% Copyright (C) 2012 by Zhen Li (zhenli3@illinois.edu).

[n, d] = size(X);
if exist('A', 'var')
    assert(isequal(size(A), [d d]) || isscaler(A));
else
    A = 1;
end

D = X * A * X';     % D(i,j) = x_i' * A * x_j
dd = diag(D);       % dd(i) = x_i' * A * x_i
D = bsxfun(@plus, bsxfun(@plus, -2*D, dd), dd');
D(sub2ind([n, n], 1:n, 1:n)) = 0;
