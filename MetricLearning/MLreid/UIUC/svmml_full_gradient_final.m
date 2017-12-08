function [f,df] = svmml_full_gradient_final (para,X,Y,lambda1,lambda2)


[n, d] = size(X);
assert(numel(Y) == n);

A = reshape(para(1:d^2),d,d);
B = reshape(para(d^2+1:end-1),d,d);
b = para(end);

% % % ensure symmetry for A and B
A = (A + A')/2;
B = (B + B')/2;

% Constrct pairwise class similarity matrix.
YY = bsxfun(@eq, Y(:), Y(:)');
np = nnz(YY) - n;       % # positive pairs (w/o diagonal terms) not counting 
                        % the pair with itself
nn = n^2 - n - np;      % # negative pairs
% create label map
Ymap = sign(YY - 0.5);

pre_f = sum((A*X').*X',1);
% pre_f = sum(Xp.^2,1);
% pre_f = diag(X*A*X');
%1. X_i'*M*M'*X_i at repmat in column wise
f1 = repmat(pre_f',[1,n]);
%1. X_i'*M*M'*X_i at repmat in row wise
f2 = f1';
%3 X_i'*N*N'*X_j
f3 = X*B*X';

g = 0.5*f1 + 0.5*f2 - f3 + b;

L = masked_rescale(Ymap,YY,1/np,1/nn);
L = abs(L);

% the reason we need add the special handle here is 
% log(exp(5000)) = Inf in matlab instead of 5000;
E = g.*Ymap;
big_idx = find(E>60);
gnew = log(exp(E)+1);
gnew(big_idx) = E(big_idx);
gnew = gnew.*L;
% take out the diagnoal effect
gnew(sub2ind([n,n],1:n,1:n)) = 0;

% calculate the value of the objective function
f = sum(gnew(:)) + 0.5*(lambda1*sum(A(:).^2)+lambda2*sum(B(:).^2));



%% calculate the gradient

% compute a matrix W (NxN)
% W = (1/C)*(1/1+e^(-l_{ij}*f(x))*l_{ij});

% here need to take out the effect of diagonal line
Ymap_remove = Ymap;
Ymap_remove(sub2ind([n,n],1:n,1:n)) = 0;

W = (1./(1+exp(-g.*Ymap))).*L.*Ymap_remove;

dA = lambda1*A + 0.5*((repmat(sum(W,2)',[d,1]).*X')*X + ...
    + (repmat(sum(W,1),[d,1]).*X')*X);

dB = lambda2.*B - X'*W*X;

db = sum(W(:));

df = [dA(:); dB(:); db];

end