function [K] = project(Ky_test,Ry,RyIndex,RySize)

% This function will project a kernel matrix (a testing by training)
% into the gram-schmidt space as defined by the original training
% kernel.
%
% Usage: 
%     [K] = project(Ky_test,Ry,RyIndex,RySize)
% Output : K - projected testing kernel
% Input  : Ky_test - the testing kernel
%          Ry, RyIndex and RySize outputs from the gsd_mex 
%          algorithm for the training kernel
%
% Written by David R. Hardoon 
% drh@ecs.soton.ac.uk

T = size(Ry,2);
N = size(Ky_test,1);
K = zeros(N,T);

for i=1:N
   for j=1:T
      temp = 0;
      for t=1:j-1
         temp = K(i,t)*Ry(RyIndex(j),t) + temp;
      end
      K(i,j) = (Ky_test(i,RyIndex(j)) - temp)/RySize(j);
   end
end

