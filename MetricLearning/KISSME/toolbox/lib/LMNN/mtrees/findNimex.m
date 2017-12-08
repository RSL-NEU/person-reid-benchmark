function [iknn]=findNImex(tree,xtrain,xtest,nitrain,nitest);
% function [iknn,dists]=findNImex(tree,xtrain,xtest,nitrain,nitest);
%
% copyright by Kilian Q. Weinberger, 2007
%

% do some basic checks:
if (size(nitest,2)~=size(xtest,2))
 error('Testing and Nitest data must have same length!');
end;
if (size(nitrain,2)~=size(xtrain,2))
 error('Training and Nitrain data must have same length!');
end;

if (size(xtrain,1)~=size(xtest,1))
 error('Training input and output must have the same dimensions!');
end;

dim=size(tree.piv,1);
[iknn]=findNimtree(xtrain(1:dim,tree.index),xtest(1:dim,:),nitrain(tree.index),nitest,tree.piv,tree.radius,tree.jumpindex,tree.kids);
try
 iknn(2,:)=tree.index(iknn(2,:));
catch
    fprintf('Something went wrong!\n');
    keyboard;
end;