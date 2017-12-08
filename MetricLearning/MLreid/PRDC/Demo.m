function Demo

% load the data
load('data.mat')

% get basic information
GallerySampleAmount = sum(CountOfGalleryTargetSamples);
TestSampleAmount = sum(CountOfTestingTargetSamples);

% parameters
option.tolerance_ratio = 10^-2;
MaxLoop = 100;
option.K = [];
option.normalise_by_variance = 0; % this is to normalise the data
option.is_auto_adjust_initial_projection = 0;% for non-normalised data, you can try to set it to 1
Dimension = 100;

% learn the ranking distance by PRDC
[ProjectionCoefficient,option,pos_neg_option,] = ...
    LogPenalizedExpRankSubpsace_Seq(...
             TrainingTargetDataSetC,CountOfTrainingTargetSamples,...
             option,MaxLoop,...
             Dimension); % ProjectionCoefficient is the returned projection vector for distance modelling

% - get the distance matrix
DistanceMatrix = zeros(GallerySampleAmount,TestSampleAmount);
for test_sample_id = 1 : TestSampleAmount
    % learn the difference vector
    [DiffSetC] = ...
        MakeDiffSubset(GalleryDataSetC,...
              TestingTargetDataSetC(:,test_sample_id),...
              pos_neg_option.is_abs_diff,1);
    DistanceMatrix(:,test_sample_id) ...
        = ...
           sum((ProjectionCoefficient' ...
               * DiffSetC) .^ 2,1)';    
end
                    
% the DistanceMatrix(i,j) is the distance value
% between the ith training sample and jth testing sample

% testing
% - get the n rank recognition rate
%% !!! We assume there is only one image for each gallery class here !!! 
[SortedDistanceMatrix,Index] = sort(DistanceMatrix);
[ExtentionArrayWithClassLabel] = StandardExtendClassLableArray(CountOfTestingTargetSamples);
if size(ExtentionArrayWithClassLabel,1) > size(ExtentionArrayWithClassLabel,2)
    ExtentionArrayWithClassLabel = ExtentionArrayWithClassLabel';
end

for rank = 1 : 20
    Temp = Index(1 : rank,:) == ones(rank,1) * ExtentionArrayWithClassLabel;    
    Rank1cAccRate = ...
        sum(sum(Temp==1,1)) / TestSampleAmount;
    disp(['MatchingRate for Rank ' num2str(rank) ' is ' num2str(Rank1cAccRate*100) ' %']);
end