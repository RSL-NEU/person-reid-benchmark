function [ ds, runs ] = CrossValidateViper(ds, learn_algs, X, idxa, idxb, params)
%function [ds,runs]=CrossValidateViper(ds,learn_algs,X,idxa,idxb,params)
%   
% Input:
%
%  ds - data struct that stores the result
%  learn_algs - algorithms that are used for cross validation
%  X - input matrix, each column is an input vector [DxN*2]. N is the
%  number of pairs.
%  idxa - index of image A in X [1xN]
%  idxb - index of image B in X [1xN]
%  params.N - number of pairs
%  params.numFolds - number of runs
%
% Output:
%
%  ds - struct [1xnumFolds] that contains the result
%  runs - struct [1xnumFolds] that contains the train test split
%  runs(c).perm - random permutation of run c
%  runs(c).idxtrain - train index
%  runs(c).idxtest - test index
%
% See also CrossValidatePairs
%
% copyright by Martin Koestinger (2011)
% Graz University of Technology
% contact koestinger@icg.tugraz.at
%
% For more information, see <a href="matlab: 
% web('http://lrs.icg.tugraz.at/members/koestinger')">the ICG Web site</a>.

runs.numFolds = params.numFolds;
for c=1:params.numFolds
    clc; fprintf('VIPeR run %d of %d\n',c,params.numFolds);

    % draw random permuation 
    perm = randperm(params.N);

    % split in equal-sized train and test sets
    idxtrain = perm(1:params.N/2);
    idxtest  = perm(params.N/2+1:end);
    
    runs(c).perm = randperm(params.N);
    runs(c).idxtrain = idxtrain;
    runs(c).idxtest = idxtrain;
   
    % train on first half
    for aC=1:length(learn_algs)
        cHandle = learn_algs{aC};
        fprintf('    training %s ',upper(cHandle.type));
        s = learnPairwise(cHandle,X,[idxa(idxtrain) idxa(idxtrain)],[idxb(idxtrain) idxb(idxtrain(randperm(params.N/2)))],logical([ones(1,size(idxtrain,2)) zeros(1,size(idxtrain,2))]));
        if ~isempty(fieldnames(s))
            fprintf('... done in %.4fs\n',s.t);
            ds(c).(cHandle.type) = s;
        else
            fprintf('... not available');
        end
    end
       
    % test on second half
    names = fieldnames(ds(c));
    for nameCounter=1:length(names)       
        fprintf('    evaluating %s ',upper(names{nameCounter}));
        ds(c).(names{nameCounter}).cmc = calcMCMC(ds(c).(names{nameCounter}).M, X,idxa,idxb,idxtest);
        fprintf('... done \n');
    end
end

end