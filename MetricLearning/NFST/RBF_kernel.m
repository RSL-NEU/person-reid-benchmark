function [K_train, K_test, mu] = RBF_kernel(X, Y)
    norm1 = sum(X.^2,2);
    norm2 = sum(X.^2,2);
    dist = (repmat(norm1 ,1,size(X,1)) + repmat(norm2',size(X,1),1) - 2*X*X');
    clear norm1 norm2
    mu=sqrt(mean(dist(:))/2);
    K_train = exp(-0.5/mu^2 * dist);

    clear dist


    norm1 = sum(X.^2,2);
    norm2 = sum(Y.^2,2);
    dist = (repmat(norm1 ,1,size(Y,1)) + repmat(norm2',size(X,1),1) - 2*X*Y');
    clear norm1 norm2
    K_test = exp(-0.5/mu^2 * dist);

    clear dist


end