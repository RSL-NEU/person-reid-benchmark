clear all
knn=3;  % we are optimizing ver the knn=3 nearest neighbors classifier
disp(['Automatic tuning of LMNN parameters for ' num2str(knn) '-NN classification.']); 

%% load data
load data/segment

%% tune parameters
[K,mu,outdim,maxiter]=findLMNNparams(xTr,yTr,knn); 

%% train full muodel
fprintf('Training final model....');
[L,Details] = lmnn2(xTr, yTr,K,'maxiter',maxiter,'quiet',1,'outdim',outdim,'mu',mu,'subsample',1.0);
testerr=knncl(L,xTr,yTr,xTe,yTe,3,'train',0);
fprintf('testing error=%2.4f\n',testerr);

