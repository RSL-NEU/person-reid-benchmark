%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) Copyright 2012-13  MICC - Media Integration and Communication Center,
% University of Florence. Giuseppe Lisanti <giuseppe.lisanti@unifi.it> and 
% Iacopo Masi <iacopo.masi@unifi.it>. Fore more details see URL 
% http://www.micc.unifi.it/lisanti/source-code/re-id
%
%
% [ Dnorm norms ]= normalizeBase(D)
% 
% The function gets the super base D and normalized each column such that
% l2-norm is 1. It returns also the norm colmun-wise.
%
% Input
% 
% D: the superbase
%
% Output
%
% Dnorm: the normalized superbase
% norms: a vector that contains the norm l2 for each column
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ Dnorm norms ]= normalizeBase(D)

%% Normalize dictionary to have unit l2-norm
norms = sqrt(sum(D.^2));
Dnorm = mexNormalize(D);

return;