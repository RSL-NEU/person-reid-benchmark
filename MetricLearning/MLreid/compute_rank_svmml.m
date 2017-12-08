function [R,Alldist,ixx] = compute_rank_svmml(Method,train,test,ix_partition, IDs)

% A = Method.A;
% B = Method.B;
% b = Method.b;
% [K_test] = ComputeKernelTest(train, test, Method); %compute the kernel matrix.
% K_test = test';

for k = 1:size(ix_partition,1)
    ix_ref = ix_partition(k,:) == 1;
    if min(min(double(ix_partition))) < 0
        ix_prob = ix_partition(k,:) ==-1; 
    else
        ix_prob = ix_partition(k,:) ==0;
    end
    ref_ID = IDs(ix_ref);
    prob_ID = IDs(ix_prob);
    
    dis = 0;
    for c = 1:numel(test)
        A = Method{c}.A;
        B = Method{c}.B;
        b = Method{c}.b;
        K_test = test{c}';
        K_ref = K_test(:, ix_ref);
        K_prob = K_test(:, ix_prob);
        
        
        max_dim = max(sum(ix_prob),sum(ix_ref));
        f1 = zeros(max_dim);
        f2 = zeros(max_dim);
        f3 = zeros(max_dim);
        f1 = 0.5*repmat(diag(K_prob'*A*K_prob),[1,sum(ix_ref)]);
        f2 = 0.5*repmat(diag(K_ref'*A*K_ref)',[sum(ix_prob),1]);       
        f3 = K_prob'*B*K_ref;
        dis = dis+f1+f2-f3+b;
    end

    for i = 1:size(K_prob,2)
        [tmp, ix] = sort(dis(i, :));
        r(i) =  find(ref_ID(ix) == prob_ID(i));
        ixx(i,:)=ix;
    end
    R(k, :) = r; 
    Alldist{k} = dis; % distance matrix
end


% Calculate the kernel matrix for train and test set.
% TODO: Replace the ComputeKernel function in  ComputeKernel.m
% Input: 
%       Method: the distance learning algorithm struct. In this function
%               only field used "kernel", the name of the kernel function. 
%       train: The data used to learn the projection matric. Each row is a
%               sample vector. Ntr-by-d
%       test: The data used to test and calculate the CMC for the
%               algorithm. Each row is a sample vector. Nts-by-d
function [K_test] = ComputeKernelTest(train, test, Method)

if (size(train,2))>2e4 && (strcmp(Method.kernel, 'chi2') || strcmp(Method.kernel, 'chi2-rbf'))
    % if the input data matrix is too large then use parallel computing
    % tool box.
    matlabpool open
    
    switch Method.kernel
        case {'linear'}
            K_test = train * test';
        case {'chi2'}
            parfor i =1:size(test,1)
                dotp = bsxfun(@times, test(i,:), train);
                sump = bsxfun(@plus, test(i,:), train);
                K_test(:,i) = 2* sum(dotp./(sump+1e-10),2);
            end
        case {'chi2-rbf'}
            sigma = Method.rbf_sigma;
            parfor i =1:size(test,1)
                subp = bsxfun(@minus, test(i,:), train);
                subp = subp.^2;
                sump = bsxfun(@plus, test(i,:), train);
                K_test(:,i) =  sum(subp./(sump+1e-10),2);
            end
            K_test =exp(-K_test./sigma);
    end
    matlabpool close
else
    switch Method.kernel
        case {'linear'}
            K_test = train * test';
        case {'chi2'}
            for i =1:size(test,1)
                dotp = bsxfun(@times, test(i,:), train);
                sump = bsxfun(@plus, test(i,:), train);
                K_test(:,i) = 2* sum(dotp./(sump+1e-10),2);
            end
        case {'chi2-rbf'}
            sigma = Method.rbf_sigma;
            for i =1:size(test,1)
                subp = bsxfun(@minus, test(i,:), train);
                subp = subp.^2;
                sump = bsxfun(@plus, test(i,:), train);
                K_test(:,i) =  sum(subp./(sump+1e-10),2);
            end
            K_test =exp(-K_test./sigma);
    end
end
return;