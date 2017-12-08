%% Mean removal + L2 norm normalization
% Require:
%
% parFea -- paramter cells of features.
%    parFea.featurenum  -- number of different type of GOG.
%    parFea.usefeature  -- whether use the specific type of GOG. Size:[parFea.featurenum, 1];
%
% feature_cell -- feature cells of trainining or test data
%                 each cell contains feature vectors of feature type f whose size is: [datanum, feature dimension]
%
% tot -- whether feature_cell is traing or test data,  1 -- training data, 2 -- test data
%
% mean_cell -- mean vectors of training data
%              (require only if tot == 2, when tot == 1 it is learned).

if tot == 1; mean_cell = cell(parFea.featurenum, 1); end

for f = 1:parFea.featurenum
    if parFea.usefeature(f) == 1
        X = feature_cell{f,1};  % X: feature vectors of feature type f. Size: [datanum, feature dimension]
        
        if tot == 1 % training data
            meanX = mean( X, 1); % meanX -- mean vector of features
            mean_cell{f} = meanX;
        end
        if tot == 2 % test data
            meanX = mean_cell{f};
        end
        
        Y = ( X - repmat(meanX, size(X,1), 1)); % Mean removal
        for dnum = 1:size(X, 1)
            Y(dnum,:) = Y(dnum, :)./norm(Y(dnum, :), 2); % L2 norm normalization
        end
           
        feature_cell{f,1} = Y;
    end
end