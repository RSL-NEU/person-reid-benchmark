function [iknn,dists]=usemtreemexomp(xtest,xtrain,tree,k);
% function [iknn,dists]=usemtree(xtest,xtrain,tree,k);
%
% copyright by Kilian Q. Weinberger, 2007
%

if(nargin<4)
 k=5;
end;

% do some basic checks:
if (size(xtest,1)~=size(xtrain,1))
 error('Training and Test data must have same dimensions!');
end;

dim=size(tree.piv,1);
[iknn,dists]=findknnmtreeomp(xtrain(1:dim,tree.index),xtest(1:dim,:),k,tree.piv,tree.radius,tree.jumpindex,tree.kids);


iknn=tree.index(iknn);
if(ndims(iknn)==1) iknn=iknn(:);end;
