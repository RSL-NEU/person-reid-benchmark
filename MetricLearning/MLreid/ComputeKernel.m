% calculate the kernel matrix.
% By Fei Xiong, 
%    ECE Dept, 
%    Northeastern University 
%    2013-11-04
% Input: 
%       Method: the distance learning algorithm struct. In this function
%               two field are used. 
%               rbf_sigma is written while computing the rbf-chi2 kernel
%               matrix.
%               kernel is the name of the kernel function. 
%       X:  Each row is a sample vector. N-by-d


function [K, Method] = ComputeKernel(X, kernel, Method)
info = whos('X');
if strfind(info.class,'gpu')
    fprintf('Using GPU now\n');
    K = gpuArray.zeros(size(X,1),'single');
else 
    K = zeros(size(X,1),'single');
end
Method.rbf_sigma = [];
if 0%isempty(strfind(info.class,'gpu'))%(numel(X))>2e4 && (strcmp(kernel, 'chi2') || strcmp(kernel, 'chi2-rbf'))
    
%     poolobj = parpool;
    switch kernel
        case {'linear'}% linear kernel
            K = X*X';
        case {'chi2'}% chi2 kernel
            parfor i =1:size(X,1)
                dotp = bsxfun(@times, X(i,:), X);
                sump = bsxfun(@plus, X(i,:), X);
                K(i,:) = full(2* sum(dotp./(sump+1e-10),2));
            end
            clear subp sump;
        case {'chi2-rbf'}% chi2 RBF kernel
            parfor i =1:size(X,1)
                subp = bsxfun(@minus, X(i,:), X);
                subp = subp.^2;
                sump = bsxfun(@plus, X(i,:), X);
                K(i,:) = full(sum(subp./(sump+eps),2));
            end
            temp = triu(ones(size(K))-eye(size(K)))>0;
            temp = K(temp(:));
            [temp,~]= sort(temp);
            % rbf-chi2 kernel parameter can be set here. For example, the
            % first quartile of the distance can be used to normalize the
            % distribution of the chi2 distance  
%             Method.rbf_sigma = 1; %temp(ceil(length(temp)*0.25));
            Method.rbf_sigma = mean(mean(temp)); %temp(ceil(length(temp)*0.25));% following KCCA
            K =exp( -K/(2*Method.rbf_sigma));
            clear subp sump;
        case {'exp'}
            parfor i = 1:size(X,1)
                diff = bsxfun(@minus,X(i,:),X);
                K(i,:) = sqrt(sum(diff.^2,2));
            end
            temp = triu(ones(size(K))-eye(size(K)))>0;
            temp = K(temp(:));
            [temp,~]= sort(temp);
            Method.rbf_sigma = temp(ceil(length(temp)*0.25));
            K = exp(-K/(2*Method.rbf_sigma^2));
        case {'poly'}
            K = (X*X'+1).^2;
    end
%     delete(poolobj);
else
    switch kernel
        case {'linear'}% linear kernel
            K = X*X';
        case {'chi2'}% chi2 kernel
            for i =1:size(X,1)
                dotp = bsxfun(@times, X(i,:), X);
                sump = bsxfun(@plus, X(i,:), X);
                K(i,:) = full(2* sum(dotp./(sump+1e-10),2));
            end
            clear subp sump;
        case {'chi2-rbf'}% chi2 RBF kernel
            for i =1:size(X,1)
                subp = bsxfun(@minus, X(i,:), X);
                subp = subp.^2;
                sump = bsxfun(@plus, X(i,:), X);
                K(i,:) = full(sum(subp./(sump+1e-10),2));
            end
            temp = triu(ones(size(K))-eye(size(K)))>0;
            temp = K(temp(:));
            [temp,~]= sort(temp);
            Method.rbf_sigma = mean(mean(temp)); %temp(ceil(length(temp)*0.25));% following KCCA
            K =exp( -K/(2*Method.rbf_sigma));
            clear subp sump;
        case {'exp'}
            for i = 1:size(X,1)
                diff = bsxfun(@minus,X(i,:),X);
                K(i,:) = sqrt(sum(diff.^2,2));
            end
            temp = triu(ones(size(K))-eye(size(K)))>0;
            temp = K(temp(:));
            [temp,~]= sort(temp);
            Method.rbf_sigma = temp(ceil(length(temp)*0.25));
            K = exp(-K/(2*Method.rbf_sigma^2));
        case {'poly'}
            K = (X*X'+1).^2;
    end
end

if strfind(info.class,'gpu')
    K = gather(K);
    Method.rbf_sigma = gather(Method.rbf_sigma);
end
return;
