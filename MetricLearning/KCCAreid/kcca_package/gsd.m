function [feat,tSize,index] = gsd(K,T)
% incomplete gram-schmidt decomposition algorithm
%
% [feat,tSize,index] = gsd(K,T)
%
% Input: K - Kernel Matrix of size lxl 
%        T - number of latent variables to be found
% 
% Output: feat  - new decomposed Matrix
%         tSize - A record of the feat division (don't really need
%                 but found useful)
%         index - The index of the modified elements
%
% David R. Hardoon, drh@ecs.soton.ac.uk
% modified by K. Veropoulos, verop@unr.edu
%
% No commercial use.
% Any modification, please email D.R. Hardoon a copy.

	% initializations...
	m = size(K,1);
	index = zeros(m,1);
	tSize = zeros(m,1);
	feat = zeros(m);
	i=1:m;
	
	% saving the diagonal into norm2
	norm2 = diag(K);
	j = 1;
	
	% running the modified gram-schimt algorithm
	while (sum(norm2) > T) && (j ~= m+1)

      % finding best new element
      [value, j2] = max(norm2);

      % saving the index
      index(j) =  j2;

      % setting
      tSize(j) = sqrt(norm2(j2));

      % calculating new features
      t=1:j-1;
      tSum = feat(i,t)*feat(j2,t)'; 
      feat(i,j) = (K(i,j2)-tSum)/tSize(j);

      % updating diagonal elements
      norm2(i) = norm2(i) - feat(i,j).^2;
	
      j = j + 1;
	end
	
	% checking to see if feat needs to be cropped
	feat = feat(1:m,1:j-1);
return;
