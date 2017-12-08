function [K_test] = RBF_kernel2(X, Y, mu)

    norm1 = sum(X.^2,2);
    norm2 = sum(Y.^2,2);
    dist = (repmat(norm1 ,1,size(Y,1)) + repmat(norm2',size(X,1),1) - 2*X*Y');
    clear norm1 norm2
    K_test = exp(-0.5/mu^2 * dist);

    clear dist


end