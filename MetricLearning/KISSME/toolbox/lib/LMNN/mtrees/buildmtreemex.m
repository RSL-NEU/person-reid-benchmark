function [tree]=buildmtreemex(x,mi)
%function [xoutput,index,treeinfo]=buildmtree(xinput,ma);
%
% input: 
% xinput :  input vectors (columns are vectors)
% ma : maximum number of points in leaf
%
% output:
%
% xoutput : input vectors reshuffled (sorted according to tree assignment)
% index :	index of xouptput (ie xoutput=xinput(x) )
% treeinfo : all the necessary information for the tree
% 
%
% copyright by Kilian Q. Weinberger, 2007
%

if(~exist('mi'))
	mi=5;
end;

[tree.index,tree.piv,tree.radius,tree.jumpindex,tree.kids]=buildmtreec(x,mi);
