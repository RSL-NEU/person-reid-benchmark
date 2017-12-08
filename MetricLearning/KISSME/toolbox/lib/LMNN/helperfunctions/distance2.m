function dist=distance2(X,x)
% dist=distance(X,x)
%
% computes the pairwise squared distance matrix between any column vectors in X and
% in x
%
% INPUT:
%
% X     dxN matrix consisting of N column vectors
% x     dxn matrix consisting of n column vectors
%
% OUTPUT:
%
% dist  Nxn matrix 
%
% Example:
% Dist=distance(X,X);
% is equivalent to
% Dist=distance(X);
%
%
% 1.0 Initial version
% 1.1 New distance function that is faster for larger matrices.
%
% $Id: distance.m 1223 2007-12-02 21:41:10Z kilianw $

[D,N] = size(X);
 if(nargin>=2)
  [d,n] = size(x);
  if(D~=d)
   error('Both sets of vectors must have same dimensionality!\n');
  end;
  dist = bsxfun(@plus, dot(x,x,1), dot(X,X,1)')-2*(X.'*x);
 else
  [D,N] = size(X);
  s=sum(X.^2,1);
  dist=bsxfun(@plus,s',bsxfun(@plus,s,-2*X.'*X));
end;

