function [option,Crieterion_Array,total_criterion_value] = ...
    LogPenalizedExpRankSubpsace_Seq(...
             train,gID, ix_pairs,label_pairs,...
             option)

% ------- input ----------------
% train: train(j,:) is the jth sample vector. 
% option: option.tolerance_ratio is the ratio between two criterion values
%         at two consecutive iterations respective. You can set is 0.0001. For
%         others, please set
%                     option.is_auto_adjust_initial_projection: let the algorithm automatically adjust the initilised value for each iteration 
%                     option.normalise_by_variance: let the algorithm to automatically normalise the data in order to avoid computational problem caused by the expectation matrix  
%                     option.InitialW_Projection: the initialsed projection assigned by user. If not given, it will be automatically initialised 
% MaxLoop: Maximum number of iterations
% Dimension: the maximum rank of the metric. Just set it large for example
%            1000, and the program would automatically estimate it.

% rearrange the features and the ID vectors 
uID = unique(gID);
DataSetC = zeros(size(train));
CountOfSampleEachClass = zeros(1,numel(uID));
cnt = 1;
for i = 1:numel(uID)
    tmp = (gID == uID(i));
    DataSetC(cnt:cnt+sum(tmp)-1,:) = train(tmp,:);
    CountOfSampleEachClass(i) = sum(tmp);
    cnt = cnt + sum(tmp);
end
DataSetC = single(DataSetC');
MaxLoop = option.Maxloop;
Dimension = option.Dimension;

% setting
is_pos_projection = 0;
option.is_abs_diff = 1;
option.NumberOfNeighbors  = [];
option.is_local_learning = 0;

% get the difference set based on the pair index
pos_pair = label_pairs > 0;
neg_pair = label_pairs < 0;
idx_pos_pair = ix_pairs(pos_pair,:);
idx_neg_pair = ix_pairs(neg_pair,:);
PosAmountEachClass = hist(double(idx_pos_pair(:,1)),unique(double(idx_pos_pair(:,1))));
NegAmountEachClass = PosAmountEachClass*option.npratio;
PosDiffSet = (train(ix_pairs(pos_pair,1),:) - train(ix_pairs(pos_pair,2),:))';
if option.is_abs_diff
    PosDiffSet = abs(PosDiffSet);
end
IntraNeighboringPairs = [];
NegDiffSet = (train(ix_pairs(neg_pair,1),:) - train(ix_pairs(neg_pair,2),:))';
if option.is_abs_diff
    NegDiffSet = abs(NegDiffSet);
end
InterNeighboringPairs = [];

% % get the difference set
% [PosDiffSet,NegDiffSet,...
%  PosAmountEachClass,NegAmountEachClass,...
%  IntraNeighboringPairs,InterNeighboringPairs] = ...
%     GetPosNegDiffSetC(DataSetC,CountOfSampleEachClass,option);

% % only use part of the negtive pairs
% startP = [0 cumsum(PosAmountEachClass(1:end-1))];
% NegDiffSet_tmp = zeros(size(NegDiffSet,1),option.npratio*sum(PosAmountEachClass));
% NegAmount_tmp = zeros(size(NegAmountEachClass));
% endP = 0;
% for i = 1:size(PosAmountEachClass,2)
%     num_neg = option.npratio*PosAmountEachClass(i);
%     NegAmount_tmp(i) = num_neg;    
%     NegDiffSet_tmp(:,endP+1:endP+num_neg) = NegDiffSet(:,startP(i)+1:startP(i)+num_neg);
%     endP = endP + num_neg;
% end
% NegDiffSet = NegDiffSet_tmp;
% NegAmountEachClass = NegAmount_tmp;

% get basic information
DiffSetC = [PosDiffSet NegDiffSet];
clear PosDiffSet;
clear NegDiffSet;
DiffSetC = double(DiffSetC);
% get sample representation
LocalWeight = 1;
[DiffCoefficientSetC] = ...
        ExtractQueryDiffPairCoefficientC(...
             PosAmountEachClass,NegAmountEachClass,...
             IntraNeighboringPairs,InterNeighboringPairs,LocalWeight);

% get the neighoring information
InterDiffPairCount = size(DiffCoefficientSetC,2);
PairAmount = size(DiffCoefficientSetC,2);
intra_inter_pair_weight_array = ...
    [(PairAmount/InterDiffPairCount) * ones(1,InterDiffPairCount)];

start_dimension = 1;
ProjectionCoefficient = [];%zeros(size(DiffSetC,1),Dimension);
ScoreDiffArray = zeros(1,size(DiffCoefficientSetC,2));
ScoreDiffArray_Exp = 0;

% clear any not used variables
clear DataSetC

if ~isfield(option,'tolerance_ratio')
    option.tolerance_ratio = 10^-3;
end

% ---- Loop for number of features ----
criterion_value_last_feature = Inf;
Crieterion_Array = [];
total_criterion_value = 0;
for feature_id = start_dimension : Dimension
% -------------------------------------
weight_each_pair = exp(ScoreDiffArray);
Crieterion_Array{feature_id} = [];
% weight_each_pair_exp = exp(ScoreDiffArray_Exp);

step_weight = 0.1;

% initialization
w_projection = [];
if isfield(option,'InitialW_Projection')
   if  ~isempty(option.InitialW_Projection)
       w_projection = option.InitialW_Projection;
   end
end

if isfield(option,'normalise_by_variance')
   if option.normalise_by_variance == 1
       DiffSetC = ...
           DiffSetC / (eps+mean(std(DiffSetC').^2).^0.5);
   end
end
if isempty(w_projection)
    w_projection = -1 * (DiffSetC * mean(DiffCoefficientSetC,2));
end

last_w_projection = w_projection;

% ------- Lopp for each feature -------
for loop_id = 1 : MaxLoop    
% -------------------------------------
%     loop_id    
    
    if loop_id == 1   
        ProjectedDiffSetC = last_w_projection' * DiffSetC;
        
        % adjust last_w_projection automatically
        if isfield(option,'is_auto_adjust_initial_projection')
            if option.is_auto_adjust_initial_projection == 1
                last_w_projection = ...
                    last_w_projection / max(max(ProjectedDiffSetC));
                ProjectedDiffSetC = ...
                    ProjectedDiffSetC / max(max(ProjectedDiffSetC));
            end            
        end
        
        last_object_value_individual = exp(...
            sum(ProjectedDiffSetC .* ProjectedDiffSetC,1) * DiffCoefficientSetC);
        last_object_value_individual = ...
            last_object_value_individual .* weight_each_pair;
        last_criterion_value = ...
            sum(log(1+last_object_value_individual) ...
                .* intra_inter_pair_weight_array) / PairAmount;
        Crieterion_Array{feature_id} = ...
            [Crieterion_Array{feature_id} last_criterion_value];
    end
       
    TempRatioArray = ...
        last_object_value_individual ...
        ./ (1 + last_object_value_individual); 
    TempRatioArray = ...
        TempRatioArray .* intra_inter_pair_weight_array;
    TempSummaryVector = ...
        DiffCoefficientSetC * TempRatioArray';
    clear TempRatioArray;
     
    delta_w_projection = ...
        (2 / PairAmount) * ...
        DiffSetC * ...
        (ProjectedDiffSetC' .* TempSummaryVector);   
    
    
    % get the length
    ls_learn_nothing = 0;
    ischanged = 0;
    while 1
        w_projection = ...
            last_w_projection - step_weight * delta_w_projection;
        if is_pos_projection == 1
            w_projection = max(w_projection,0);
            if sum(w_projection(:)) <= 10^-6
                step_weight = step_weight * 0.5;
                continue;
            end
        end
                
        % for calculating: current_criterion_individual_value = w_projection' * DiffKernelMatrix;
        ProjectedDiffSetC = w_projection' * DiffSetC;
        current_object_value_individual = exp(...
            sum(ProjectedDiffSetC .* ProjectedDiffSetC,1) ...
            * DiffCoefficientSetC);
        
        current_object_value_individual = ...
            current_object_value_individual .* weight_each_pair;
        current_criterion_value = ...
            sum(log(1+current_object_value_individual) ...
                .* intra_inter_pair_weight_array) / PairAmount;

        if current_criterion_value < last_criterion_value
            if ischanged == 0
                step_weight = step_weight * 2;
            end
            break;
        else
            step_weight = step_weight * 0.5;
            ischanged = 1;
            if step_weight < (10^-10)
                ls_learn_nothing = 1;
                break;
            end
        end
    end

    if ls_learn_nothing == 1 % mean we cannot find any vector that can reduce the cost function
        disp(['no more update! final loop ' num2str(loop_id)])
        break;
    end
    
    last_object_value_individual = current_object_value_individual;

    Crieterion_Array{feature_id} = [Crieterion_Array{feature_id} current_criterion_value];
    if abs(current_criterion_value - last_criterion_value) < (option.tolerance_ratio * last_criterion_value)
%         current_criterion_value
%         last_criterion_value
        diff = abs(current_criterion_value - last_criterion_value);
        disp(['terminate at loop ' num2str(loop_id) '_, diff:' num2str(diff)])
        last_w_projection = w_projection;
        break;
    else
        last_w_projection = w_projection;
    end
        
    last_criterion_value = current_criterion_value;    
% ------- Loop for each feature -------
end
% -----------------------------

% return
if (ls_learn_nothing == 1) & (feature_id > 1)
    break;
else
    ProjectionCoefficient(:,feature_id) = last_w_projection;
%     break;
end
ProjectionCoefficient(:,feature_id) = last_w_projection;
if (criterion_value_last_feature <= last_criterion_value)  & (feature_id > 1)
    ProjectionCoefficient(:,feature_id : end) = [];
    break;
end
if ((criterion_value_last_feature-(10^-6)) < last_criterion_value)
    if feature_id < size(ProjectionCoefficient,2)
        disp('Learn enough!')
        %ProjectionCoefficient(:,(feature_id+1) : end) = [];
    end
    break;
end
criterion_value_last_feature = last_criterion_value;

% update the weight for each pair    
ScoreDiffArray = ...
    ScoreDiffArray ...
    + sum((last_w_projection' * DiffSetC).^ 2,1) * DiffCoefficientSetC;
% ScoreDiffArray_Exp = ...
%     ScoreDiffArray_Exp + ...
%     (sum((last_w_projection' * PosDiffSetMean).^2,1) ...
%      - sum((last_w_projection' * NegDiffSetMean).^2,1));

% project the data set to the nullspace of the existing feature space
temp_last_w_projection = last_w_projection / norm(last_w_projection);
DiffSetC = DiffSetC - temp_last_w_projection * (temp_last_w_projection' * DiffSetC);

total_criterion_value = total_criterion_value + last_criterion_value;

% ---- Loop for number of features ----
end
% -------------------------------------
option.P = ProjectionCoefficient;
option.name = 'PRDC';

if feature_id == Dimension
    disp('Learn all features!')
end