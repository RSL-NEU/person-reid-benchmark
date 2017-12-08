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
info = whos('test');
if ~strfind(info.class,'gpu')%(numel(train))>2e4 && (strcmp(Method.kernel, 'chi2') || strcmp(Method.kernel, 'chi2-rbf'))
    % if the input data matrix is too large then use parallel computing
    % tool box.
%     poolobj = parpool;
    
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
            K_test =exp(-K_test./(2*sigma));
        case {'exp'}
            parfor i = 1:size(test,1)
                diff = bsxfun(@minus,test(i,:),train);
                K_test(:,i) = sqrt(sum(diff.^2,2));
            end
            K_test = exp(-K_test/(2*Method.rbf_sigma^2));
        case {'poly'}
            K_test = (train*test'+1).^2;

    end
%     delete(poolobj)
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
            K_test =exp(-K_test./(2*sigma));
        case {'exp'}
            for i = 1:size(test,1)
                diff = bsxfun(@minus,test(i,:),train);
                K_test(:,i) = sqrt(sum(diff.^2,2));
            end
            K_test = exp(-K_test/(2*Method.rbf_sigma^2));
        case {'poly'}
            K_test = (train*test'+1).^2;
    end
end
return;