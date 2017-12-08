function W=LDA(galX, probX, galLabels, probLabels)
%% This dimensionality reduction method is a practical computation of cross-view subspace learning.
% If you find it useful, please kindly cite the following paper.
% Reference:
%   Shengcai Liao, Yang Hu, Xiangyu Zhu, and Stan Z. Li. Person
%   re-identification by local maximal occurrence representation and metric
%   learning. In IEEE Conference on Computer Vision and Pattern Recognition, 2015.
% Author: Shengcai Liao
% Institute: National Laboratory of Pattern Recognition,
% Institute of Automation, Chinese Academy of Sciences

lambda = 0.001;
qdaDims = -1;
verbose = false;

if nargin >= 5 && ~isempty(options)
    if isfield(options,'lambda') && ~isempty(options.lambda) && isscalar(options.lambda) && isnumeric(options.lambda)
        lambda = options.lambda;
    end
    if isfield(options,'qdaDims') && ~isempty(options.qdaDims) && isscalar(options.qdaDims) && isnumeric(options.qdaDims) && options.qdaDims > 0
        qdaDims = options.qdaDims;
    end
    if isfield(options,'verbose') && ~isempty(options.verbose) && isscalar(options.verbose) && islogical(options.verbose)
        verbose = options.verbose;
    end
end

if verbose == true
    fprintf('options.lambda = %g.\n', lambda);
    fprintf('options.qdaDims = %d.\n', qdaDims);
    fprintf('options.verbose = %d.\n', verbose);
end

[numGals, d] = size(galX); % n
numProbs = size(probX, 1); % m

% If d > numGals + numProbs, it is not necessary to apply XQDA on the high dimensional space. 
% In this case we can apply XQDA on QR decomposed space, achieving the same performance but much faster.
if d > numGals + numProbs
    if verbose == true
        fprintf('\nStart to apply QR decomposition.\n');
    end
    
    t0 = tic;
    [W, X] = qr([galX', probX'], 0); % [d, n]
    galX = X(:, 1:numGals)';
    probX = X(:, numGals+1:end)';
    d = size(X,1);
    clear X;
    
    if verbose == true
        fprintf('QR decomposition time: %.3g seconds.\n', toc(t0));
    end
end


labels = unique([galLabels; probLabels]);
c = length(labels);

if verbose == true
    fprintf('#Classes: %d\n', c);
    fprintf('Compute intra/extra-class covariance matrix...');
end
    
t0 = tic;

galW = zeros(numGals, 1);
galClassSum = zeros(c, d);
probW = zeros(numProbs, 1);
probClassSum = zeros(c, d);
ni = 0;

for k = 1 : c
    galIndex = find(galLabels == labels(k));
    nk = length(galIndex);
    galClassSum(k, :) = sum( galX(galIndex, :), 1 );
    
    probIndex = find(probLabels == labels(k));
    mk = length(probIndex);
    probClassSum(k, :) = sum( probX(probIndex, :), 1 );
    
    ni = ni + nk * mk;
    galW(galIndex) = sqrt(mk);
    probW(probIndex) = sqrt(nk);
end

galSum = sum(galClassSum, 1);
probSum = sum(probClassSum, 1);
galCov = galX' * galX;
probCov = probX' * probX;

galX = bsxfun( @times, galW, galX );
probX = bsxfun( @times, probW, probX );
inCov = galX' * galX + probX' * probX - galClassSum' * probClassSum - probClassSum' * galClassSum;
exCov = numProbs * galCov + numGals * probCov - galSum' * probSum - probSum' * galSum - inCov;

ne = numGals * numProbs - ni;
inCov = inCov / ni;
exCov = exCov / ne;

inCov = inCov + lambda * eye(d);

if verbose == true
    fprintf(' %.3g seconds.\n', toc(t0));
    fprintf('#Intra: %d, #Extra: %d\n', ni, ne);
    fprintf('Compute eigen vectors...');
end


t0 = tic;
[V, S] = svd(inCov \ exCov);

if verbose == true
    fprintf(' %.3g seconds.\n', toc(t0));
end

latent = diag(S);
[latent, index] = sort(latent, 'descend');
energy = sum(latent);
minv = latent(end);

weight=cumsum(latent./sum(latent));
r=sum(weight <0.95);  
% r = sum(latent >1);  
energy = sum(latent(1:r)) / energy;   

if qdaDims > r
    qdaDims = r;
end
 
if qdaDims <= 0
    qdaDims = max(1,r);
end

if verbose == true
    fprintf('Energy remained: %f, max: %f, min: %f, all min: %f, #opt-dim: %d, qda-dim: %d.\n', energy, latent(1), latent(max(1,r)), minv, r, qdaDims);
end

V = V(:, index(1:qdaDims));
if ~exist('W', 'var');
    W = V;
else
    W = W * V;
end
 W=W';
end