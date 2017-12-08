% Modified version from David R. Hardoon
%
% http://www.davidroihardoon.com/Professional/Code_files/cca.m
%
% @article{hardoon:cca,
% author = {Hardoon, David and Szedmak, Sandor and {Shawe-Taylor}, John},
% title = {Canonical Correlation Analysis: An Overview with Application to Learning Methods},
% booktitle = {Neural Computation},
% volume = {Volume 16 (12)},
% pages = {2639--2664},
% year = {2004} }

function [Wx, Wy, r] = cca(X,Y,k)

% CCA calculate canonical correlations
%
% [Wx Wy r] = cca(X,Y) where Wx and Wy contains the canonical correlation
% vectors as columns and r is a vector with corresponding canonical
% correlations.
%
% Update 31/01/05 added bug handling.

% if (nargin ~= 2)
%   disp('Inocorrect number of inputs');
%   help cca;
%   Wx = 0; Wy = 0; r = 0;
%   return;
% end


% calculating the covariance matrices
z = [X; Y];
C = cov(z.');
sx = size(X,1);
sy = size(Y,1);
Cxx = C(1:sx, 1:sx) + k*eye(sx);
Cxy = C(1:sx, sx+1:sx+sy);
Cyx = Cxy';
Cyy = C(sx+1:sx+sy,sx+1:sx+sy) + k*eye(sy);

%calculating the Wx cca matrix
Rx = chol(Cxx);
invRx = inv(Rx);
Z = invRx'*Cxy*(Cyy\Cyx)*invRx;
Z = 0.5*(Z' + Z);  % making sure that Z is a symmetric matrix
[Wx,r] = eig(Z);   % basis in h (X)
r = sqrt(real(r)); % as the original r we get is lamda^2
Wx = invRx * Wx;   % actual Wx values

% calculating Wy
Wy = (Cyy\Cyx) * Wx; 

% by dividing it by lamda
Wy = Wy./repmat(diag(r)',sy,1);

