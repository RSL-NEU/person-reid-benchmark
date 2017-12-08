% KERNEL_EXPCHI2: Compute an Exponential Kernel using Chi-Square
% distances.
%
% Usage:  [D,md] = kernel_expchi2(X,Y,[omega])
%
% Input:  X,Y are the two set of features; omega is the parameter of the 
%         exp kernel (if not specified omega is computed as the mean of the
%         distances among the training examples)
% Output: D is the Kernel matrix; md is the mean of the distances among the
%         training examples
% 
% written by Lamberto Ballan (lamberto.ballan@unifi.it)
% University of Florence, 11/05/2013

function [D,md] = kernel_expchi2(X,Y,omega)
    
  D = zeros(size(X,1),size(Y,1));
  parfor i=1:size(Y,1)
    d = bsxfun(@minus, X, Y(i,:));
    s = bsxfun(@plus, X, Y(i,:));
    D(:,i) = sum(d.^2 ./ (s+eps), 2);
  end
	
  md = mean(mean(D));
  
  if nargin < 3
    omega = md;
  end
	
  D = exp( - 1/(2*omega) .* D);
  
end
