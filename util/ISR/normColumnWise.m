%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) Copyright 2012-13  MICC - Media Integration and Communication Center,
% University of Florence. Giuseppe Lisanti <giuseppe.lisanti@unifi.it> and 
% Iacopo Masi <iacopo.masi@unifi.it>. Fore more details see URL 
% http://www.micc.unifi.it/lisanti/source-code/re-id
%
%
% n = normColumnWise(residual,p)
% 
% The function computes the norm column wise of a given matrix.
%
% Input
%
% residual: the matrix whic the norm should be computed.
% p: the kind of norm p==2 is l2, p==1 is l-1.
%
% Output
%
% n: the norm vector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function n = normColumnWise(residual,p)
n  = sum(residual.^p,1).^(1/p);
n  = n';
return