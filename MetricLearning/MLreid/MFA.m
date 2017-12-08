% implemented according to  CVPR2013: Graph Embedding and Extensions: A General
% Framework for Dimensionality Reduction
% By Fei Xiong, 
%    ECE Dept, 
%    Northeastern University 
%    2014-02-15
% INPUT
%   X: N-by-d data matrix. Each row is a sample vector.
%   id: (N-by-1) the identification number for each sample
%   option: algorithm options
%       beta: the parameter of regularizing S_b
%       d: the dimensionality of the projected feature
%       eps: the tolerence value.
%       NNw: number of samples used for within class (0 for maximum possible)
%       NNb: number of samples used for between class
% Note that the kernel trick is used here. 
% OUTPUT
%   Method: the structure contains the learned projection matrix and
%       algorithm parameters setup.
%   V: the eigenvalues
function [Method, V]= MFA(X, id, option)
T =[];
V =[];
X = single(X);
display(['begin MFA ' option.kernel]);
beta= option.beta;
d = option.d;
eps = option.epsilon;
% compute the kernel matrix
Method = struct('rbf_sigma',0);
if gpuDeviceCount > 0 && any(size(X)>1e4) % use GPU for really large matrix
    try 
        X = gpuArray(X);
        [K, Method] = ComputeKernel(X, option.kernel, Method);
    catch
        X = gather(X);
        disp('Compute on GPU failed, using CPU now...');
        [K, Method] = ComputeKernel(X, option.kernel, Method);
    end
    reset(gpuDevice()); % reset GPU memory if used
else 
    [K, Method] = ComputeKernel(X, option.kernel, Method);
end
K=double(K);
[Ww, Wb] = MFAAffinityMatrix(K, id, option.Nw,option.Nb); % compute W, Wp in equation 13 and 14
Ew = diag(sum(Ww)) - Ww; 
Eb = diag(sum(Wb)) - Wb; 

Sw = K*Ew*K; % equation 13
Sb = K*Eb*K; % equation 14
Sw =(1-beta)*Sw + beta*trace(Sw)*eye(size(Sw))/size(Sw,1);
[T, V]= eigs(Sb, Sw, d);
T =T';

options.intraK = option.Nw;
% options.interK = option.Nb;
% options.Regu = 1;
% options.ReducedDim = d;
% options.ReguAlpha = 0.1;
% [T, ~] = MFA_CDeng(id, options, K');
% T = T';

Method.name = 'MFA';
Method.P=T;
Method.kernel=option.kernel;
Method.Prob = [];
% Method.Dataname = option.dataname;
Method.Ranking = [];
Method.Dist = [];
Method.Trainoption=option;
return;

function [Ww, Wb] = MFAAffinityMatrix(K, id, NNw,NNb)
% compute distance in the kernel space using kernel matrix
temp = repmat(diag(K), 1, size(K,1));
dis = temp + temp' - 2*K;
dis(sub2ind(size(dis), [1:size(dis,1)], [1:size(dis,1)]))=inf;
temp = repmat(id.^2, 1, length(id));
idm = temp + temp' - 2*id*id';

disw = dis;
disw(idm~=0) = inf;
[temp, ixw]= sort(disw);
ixw(isinf(temp))=0;

if NNw==0 % Use the maximum possible number of within class
    NNw = max(sum(~isinf(temp)));
end

ixw = ixw(1:NNw, :);
ixtmp = repmat([1:size(K,1)], NNw, 1);
ixtmp= ixtmp(ixw(:)>0);
ixw = ixw(ixw(:)>0);
ixtmp = sub2ind(size(K), ixtmp(:), ixw(:));
Ww = zeros(size(K));
Ww(ixtmp) = 1;
Ww = Ww+ Ww';
Ww = double(Ww>0);

disb = dis;
disb(idm==0) = inf;
[temp, ixb]= sort(disb);
ixb(isinf(temp))=0;
if NNb == 0 % use the maximum possible number of between classes
    NNb = max(sum(~isinf(temp),1));
end
ixb = ixb(1:NNb, :);
ixtmp = repmat([1:size(K,1)], NNb, 1);
ixtmp= ixtmp(ixb(:)>0);
ixb = ixb(ixb(:)>0);
ixtmp = sub2ind(size(K), ixtmp(:), ixb(:));
ixtmp= ixtmp(ixtmp>0);
Wb = zeros(size(K));
Wb(ixtmp) =1;
Wb = Wb+ Wb';
Wb = double(Wb>0);
return;

% equation 1 and 2 from paper: "Self-Tuning Spectral Clustering"
% INPUT
%   K: KernelMatrix
%   NN: the index of the nearest neighbors used to scaled the distance.
% OUTPUT
%   A: Affinity matrix
function [A] = LocalScalingAffinity(K, NN)

% compute distance in the kernel space using kernel matrix
temp = repmat(diag(K), 1, size(K,1));
dis = temp + temp' - 2*K;

[disK, ~]= sort(dis);
disK = sqrt(disK(NN+1,:));
disK = disK' * disK;
A = exp(-(dis./disK));
A = A-diag(diag(A));
return;