function [l,c]=count(a);
%  
% function [l,c]=count(a);  
%
% INPUT: 
%
% a : sorted list of doubles
%
% OUTPUT:
%
% l : sorted list of doubles with only unique elements
% c : count how many times each element appears
%
% EXAMPLE:
%
% [l,c]=count([1 1 2 2 3 3 3 4 5 5]);
%
% l=1 2 3 4 5
% c=2 2 3 1 2 
%
%
% copyright 2005 by Kilian Q. Weinberger
% University of Pennsylvania
% kilianw@seas.upenn.edu
% ********************************************

fprintf('You should not execute count.m - please make sure the compiled mex functions are in the path.');
error