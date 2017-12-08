function [f,df] = svmml_projection_gradient_final (para,X,Y,p,lambda1,lambda2)


[n, d] = size(X);
assert(numel(Y) == n);

M = reshape(para(1:d*p),d,p);
N = reshape(para(d*p+1:end-1),d,p);
b = para(end);

Ap = M*M';
Bp = N*N';
Xp = M'*X';


% Constrct pairwise class similarity matrix.
YY = bsxfun(@eq, Y(:), Y(:)');
np = nnz(YY) - n;       % # positive pairs (w/o diagonal terms) not counting 
                        % the pair with itself
nn = n^2 - n - np;      % # negative pairs
% create label map
Ymap = sign(YY - 0.5);

% calculate 
pre_f = sum(Xp.^2,1);
%1. X_i'*M*M'*X_i at repmat in column wise
f1 = repmat(pre_f',[1,n]);
%1. X_i'*M*M'*X_i at repmat in row wise
f2 = f1';
%3 X_i'*N*N'*X_j
f3 = X*Bp*X';
g = 0.5*f1 + 0.5*f2 - f3 + b;

L = masked_rescale(Ymap,YY,1/np,1/nn);
L = abs(L);

% the reason we need add the special handle here is 
% log(exp(5000)) = Inf in matlab instead of 5000;
E = g.*Ymap;
big_idx = find(E>50);
gnew = log(exp(E)+1);
gnew(big_idx) = E(big_idx);
gnew = gnew.*L;
% take out the diagnoal effect
gnew(sub2ind([n,n],1:n,1:n)) = 0;
% gnew = gnew.*L;

%% calculate the gradient

% compute a matrix W (NxN)
% W = (1/C)*(1/1+e^(-l_{ij}*f(x))*l_{ij});

% here need to take out the effect of diagonal line
Ymap_remove = Ymap;
Ymap_remove(sub2ind([n,n],1:n,1:n)) = 0;

W = (1./(1+exp(-g.*Ymap))).*L.*Ymap_remove;

dM = lambda1*M + ((repmat(sum(W,2)',[d,1]).*X')*X + ...
    + (repmat(sum(W,1),[d,1]).*X')*X)*M;

dN = lambda2.*N - 2*X'*W*X*N;

db = sum(W(:));

df = [dM(:); dN(:); db];

%% calculate the value of the objective function

f = sum(gnew(:)) + 0.5*(lambda1*sum(M(:).^2)+lambda2*sum(N(:).^2));

end
