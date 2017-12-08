%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) Copyright 2012-13  MICC - Media Integration and Communication Center,
% University of Florence. Giuseppe Lisanti <giuseppe.lisanti@unifi.it> and 
% Iacopo Masi <iacopo.masi@unifi.it>. Fore more details see URL 
% http://www.micc.unifi.it/lisanti/source-code/re-id
%
%
% residual = computeResidualColumnWise(eachfeaturesTest,idPersons,...
%   Dnorm,Alphaband)
% 
% The function computes the residual between the feature probe and the 
% reconstructed one. Eq.7.
%
% Input
%
% eachfeaturesTest: the current feature test to rank
% Dnorm: the normalized superbase (gallery set)
% Alphaband: the current reconstructed coefficient from which to start, 
% banded for that person. It refers to alpha|_i (alpha resctrited to i)
% in our paper.
% idPersons: the id for each person
%
% Output
%
% residual: the residual (error) matrix column-wise.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function residual = computeResidualColumnWise(eachfeaturesTest,idPersons,...
   Dnorm,Alphaband)
residual = repmat( eachfeaturesTest,1,length(idPersons) ) ...
   - (Dnorm*Alphaband);
return