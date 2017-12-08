function [ctrain,ctest] = centerTrainTestKernels(trkernel,tskernel)

% [ctrain,ctest] = centerTrainTestKernels(trkernel,tskernel) : centers
% train/test kernels.


l = size(trkernel,1);
j = ones(l,1);
ctrain = trkernel - (j*j'*trkernel)/l - (trkernel*j*j')/l + ((j'*trkernel*j)*j*j')/(l^2);

if( nargin > 1 )
    tk =  (1/l)*sum(trkernel,1); % (1 x l)
    tl = ones(size(tskernel,1),1); % (n x 1)
    ctest = tskernel - ( tl * tk); % ( n x l )
    tk = (1/(size(tskernel,2)))*sum(ctest,2); % ( n x 1 )   
    ctest = ctest - (tk * j'); % ( n x l )
end