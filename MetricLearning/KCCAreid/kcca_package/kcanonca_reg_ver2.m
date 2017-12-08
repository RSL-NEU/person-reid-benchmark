function [nalpha, nbeta, r, Kx, Ky] = kcanonca_reg_ver2(Kx,Ky,eta,kapa,sl,nor,Rx,Ry)
% KCCA function - regularized with kapa
%
% Performes Kernel Canoncial Correlation Analysis as a single symmetric eigenvalue problem with
% parital gram schmidt kernel decomposition
%
% Usage:
%   [nalpha, nbeta, r, Kx_c, Ky_c] =  kcanonca_reg_ver2(Kx,Ky,eta,kapa)
%   [nalpha, nbeta, r, Kx_c, Ky_c] =  kcanonca_reg_ver2(Kx,Ky,eta,kapa,sl)
%   [nalpha, nbeta, r, Kx_c, Ky_c] =  kcanonca_reg_ver2(Kx,Ky,eta,kapa,sl,nor)
%   [nalpha, nbeta, r]             =  kcanonca_reg_ver2(Kx,Ky,eta,kapa,sl,nor,Rx,Ry)
%
% Input: 
%   Kx, Ky - Kernel matrices corresponding to the two views,
%   eta    - precision parameter for gsd
%   kapa   - regularsation parameter (value 0 to 1)  
%   sl     - 1 | 0 save output from gsd method (default 1)
%   nor    - normalise features (defualt 2) 
%              0 - No normalisation and projection to kernel space
%              1 - No normalisation and and leaving GS projected space
%              2 - Normalisation and projection back to Kernel space
%              3 - Normalisation and leaving in GS projected space
%   Rx, Ry - predecomposed Kx and Ky
%   flag   - true | false, whether to run the gram-schmidt algorithm on Kx Ky
%
% Output:
%   nalpha, nbeta - contains the canonical correlation vectors as columns 
%   rc            - is a  vector with corresponding canonical correlations.
%   Kx_c, Ky_c    - The centered kernels
%
% Non commercial use.
% Written by David R. Hardoon while at University of Southampton ISIS Group, Southampton UK.
% Current email - D.Hardoon@cs.ucl.ac.uk
%
% Updated 15/12/2004 - just tidying things up
% Updated 17/12/2004 - Added control over whether to normalise or not
% Updated 28/01/2005 - when not normalising r is now a vector rather then a matrix
%                    - Also added more options normalisation and proejction (see above)
% Updated 01/02/2005 - I had case 1|0 for the nor paramtere reveresed
% Updated 06/01/2006 - Added kernel centralising
% Updated 31/04/2008 - Added output for kernels when centered
% Updated 14/03/2009 - Added sorting to the computed components


% instead of gsd one could use icd

% checking correct number of inputs
if (nargin < 4 | nargin > 8 | nargin == 7)
  disp('Incorrect number of inputs')
  nalpha = 0; nbeta = 0; r = 0;
  help kcanonca_reg_ver2
  return
end

% Setting defaults
if (nargin < 5)
  sl = 1;  
  nor = 2;
elseif (nargin < 6)
  nor = 2;
end

% need to decompose kernels
if (nargin < 7)
  % If you centre outside comment out this bit
  disp('Centering Kx and Ky');
  l = size(Kx,1);
  j = ones(l,1);
  Kx = Kx - (j*j'*Kx)/l - (Kx*j*j')/l + ((j'*Kx*j)*j*j')/(l^2);

  l = size(Ky,1);
  j = ones(l,1);
  Ky = Ky - (j*j'*Ky)/l - (Ky*j*j')/l + ((j'*Ky*j)*j*j')/(l^2);

  disp('Decomposing Kernel with PGSO');
  [Rx, RxSize, RxIndex] = gsd(Kx,eta);
  [Ry, RySize, RyIndex] = gsd(Ky,eta);

  if (sl == 1)
      disp('Saving PGSO as cca_feat_data');
      save('cca_feat_data','Rx','RxSize','RxIndex','Ry','RySize','RyIndex');
  end
else
  disp('Did you centre your data?');  
end

%disp('Creating new TxT matrix Z from MxT matrix R');
Zxx = Rx'*Rx;
Zxy = Rx'*Ry;
Zyy = Ry'*Ry;
Zyx = Zxy';
tEyeY = eye(size(Zyy));
tEyeX = eye(size(Zxx));

%disp('Computing nalpha eigenproblem');
B = (1-kapa)*tEyeX*Zxx+kapa*tEyeX;
S = chol(B)';
invS = inv(S);
A = invS*Zxy*inv((1-kapa)*tEyeY*Zyy+kapa*tEyeY)*Zyx*invS';
A = 0.5*(A'+A)+eye(size(A,1))*10e-6;
[pha,rr] = eig(A);

%disp('Sorting Output of nalpha');
r = sqrt(real(rr)); % as the original r we get is lamda^2
alpha = invS'*pha; % as \hat{\alpha} = S'*\alpha - this find alpha tidal 

% if you do not do the following line it means you will need to project your
% testing data into the Gram-Shmidt space.
invRx = Rx*inv(Rx'*Rx);
nalpha = invRx*alpha;

disp('Computing nbeta from nalpha');
% computing beta -- but as we cant comput the original beta 
% we can only compute beta twidel and not the original beta 
beta = inv((1-kapa)*tEyeY*Zyy+kapa*tEyeY)*Zyx*alpha;
t = size(Zyy,1);
beta = beta./repmat(diag(r)',t,1);

% again if you do not do the following line you must project your 
% corresponding testing examples to gsd space
invRy = Ry*inv(Ry'*Ry);
nbeta = invRy*beta; 

% normalising the feature vectors and recomputing the correlation
switch nor
case 1   % no normalisation but using GS projected space
    nbeta = beta;
	nalpha = alpha;
	r = diag(r);
	disp('WARNING: Remember to project your testing Kernels into the GS space!');
case 2   % normalisation and kernel space
	disp('Normalising compute features');

    nalpha = normalisekcca(nalpha,Kx);
	nbeta = normalisekcca(nbeta,Ky);
	r = diag(nalpha'*Kx'*Ky*nbeta);

case 3   % normalisation in the GS space
    disp('Normalising compute features');
 
    nalpha = normalisekcca(alpha,Rx);
	nbeta = normalisekcca(beta,Ry);
	r = diag(nalpha'*Rx'*Ry*nbeta);

	disp('WARNING: Remember to project your testing Kernels into the GS space!');
otherwise % nothing
    r = diag(r);
end

% Make sure the components are in order of magnitude
[r, i] = sort(r);
nalpha = nalpha(:,i);
nbeta =	nbeta(:,i);
