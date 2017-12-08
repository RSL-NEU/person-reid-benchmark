% implemented according to  CVPR2013: Local Fisher Discriminant Analysis
% for Pedestrian Re-identification.
% By Fei Xiong, 
%    ECE Dept, 
%    Northeastern University 
%    2013-11-04
% INPUT
%   X: N-by-d data matrix. Each row is a sample vector.
%   id: the identification number for each sample
%   option: algorithm options
%       beta: the parameter of regularizing S_b
%       d: the dimensionality of the projected feature
%       eps: the tolerence value.
% Note that the kernel trick is used here. 
% OUTPUT
%   Method: the structure contains the learned projection matrix and
%       algorithm parameters setup.
%   V: the eigenvalues
function [Method, V]= kLFDA(X, id, option)

T =[];
V =[];
X = single(X);
display(['begin LFDA ' option.kernel]);
beta= option.beta;
d = option.d;
eps = option.epsilon;
% compute the kernel matrix
Method = struct('rbf_sigma',0);
% Method.Kmat = option.Kmat;
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

[Aff] = LocalScalingAffinity(K, option.LocalScalingNeighbor);
temp = repmat(id.^2, 1, length(id));
temp = temp + temp' - 2*id*id';
Aw = zeros(size(temp));
Aw(temp ==0) =1;
nc = sum(Aw);
Aw = bsxfun(@times, Aw, 1./nc); %equation 10
Aw(temp ==0) = Aff(temp ==0) .* Aw(temp ==0) ;%equation 10
Ab = zeros(size(temp));
Ab(temp ==0) =1;
Ab= bsxfun(@times, Ab, -1./nc);
Ab= 1/length(id) + Ab; %equation 11 
Ab(temp ==0) = Aff(temp ==0) .* Ab(temp ==0) ;%equation 11

Ew = zeros(size(Aff));
Eb = Ew;
Ew = diag(sum(Aw)) - Aw; % part of equation 8 Ew = sum(Aw_ij*(ei-ej)*(ei-ej)')
Eb = diag(sum(Ab)) - Ab; % part of equation 9 Eb = sum(Ab_ij*(ei-ej)*(ei-ej)')

Sw = K*Ew*K; % equation 8 (1/2 can be canceled)
Sb = K*Eb*K; % equation 9 (1/2 can be canceled)
Sw =(1-beta)*Sw + beta*trace(Sw)*eye(size(Sw))/size(Sw,1);

opt.p = min(2*d,size(Sb,1));
[T, V]= eigs(Sb, Sw, d,'lm',opt);
% [T, V]= eigs(Sb, Sw, d);
T =T';
Method.name = 'LFDA';
Method.P=T;
Method.kernel=option.kernel;
Method.Prob = [];
% Method.Dataname = option.dataname;
Method.Ranking = [];
Method.Dist = [];
Method.Trainoption=option;
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
% For a little bit better result (MAYBE)
% A = exp(-sqrt(dis./disK)); 
A = A-diag(diag(A));
return;