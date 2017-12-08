function proj_null = NFST(K, labels)

    classes = unique(labels);

    %% check kernel matrix 
    [n,m] = size(K);
    if n ~= m
        error('kernel matrix must be quadratic');
    end

    %%% calculate weights of orthonormal basis in kernel space 
    centeredK = centerKernelMatrix(K);
    if issparse(centeredK)
      [basisvecs,basisvecsValues] = eig(full(centeredK));
    else
      [basisvecs,basisvecsValues] = eig(centeredK);
    end
    basisvecsValues = diag(basisvecsValues);
    basisvecs = basisvecs(:,basisvecsValues > 1e-12);
    basisvecsValues = basisvecsValues(basisvecsValues > 1e-12);
    basisvecsValues = diag(1./sqrt(basisvecsValues));
    basisvecs = basisvecs*basisvecsValues;
      
    %%% calculate transformation T of within class scatter Sw: 
    %%% T= B'*Sw*B = H*H'  and H = B'*K*(I-L) and L a block matrix
    L = zeros(n,n);
    for i=1:length(classes)

       L(labels==classes(i),labels==classes(i)) = 1/sum(labels==classes(i));

    end

    %%% need Matrix M with all entries 1/m to modify basisvecs which allows usage of 
    %%% uncentered kernel values:  (eye(size(M))-M)*basisvecs
    M = ones(m,m)/m;
    
    %%% compute helper matrix H
    H = ((eye(size(M))-M)*basisvecs)'*K*(eye(size(K))-L);
    
    %%% T = H*H' = B'*Sw*B with B=basisvecs
    T = H*H';
    
    %%%%calculate weights for null space 
    eigenvecs = null(T);
    if size(eigenvecs,2) < 1
      
      [eigenvecs, eigenvals] = eig(T);
      eigenvals = diag(eigenvals);
      [min_val min_ID] = min(eigenvals);
      eigenvecs = eigenvecs(:,min_ID);
      
    end
    
    %%% calculate null space projection 
    proj_null = ((eye(size(M))-M)*basisvecs)*eigenvecs;
    
end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function centeredKernelMatrix = centerKernelMatrix(kernelMatrix)
% centering the data in the feature space only using the (uncentered) Kernel-Matrix
%
% INPUT: 
%       kernelMatrix -- uncentered kernel matrix
% OUTPUT: 
%       centeredKernelMatrix -- centered kernel matrix

  % get size of kernelMatrix
  n = size(kernelMatrix, 1);

  % get mean values of each row/column
  columnMeans = mean(kernelMatrix); % NOTE: columnMeans = rowMeans because kernelMatrix is symmetric
  matrixMean = mean(columnMeans);

  centeredKernelMatrix = kernelMatrix;

  for k=1:n

    centeredKernelMatrix(k,:) = centeredKernelMatrix(k,:) - columnMeans;
    centeredKernelMatrix(:,k) = centeredKernelMatrix(:,k) - columnMeans';

  end

  centeredKernelMatrix = centeredKernelMatrix + matrixMean;

end 