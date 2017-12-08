function [DiffCoefficientSetC] = ...
    ExtractQueryDiffPairCoefficientC(...
       PosAmountEachClass,NegAmountEachClass,...
       IntraNeighboringPairs,InterNeighboringPairs,weight)


% get basic information
ClassAmount = length(PosAmountEachClass);
SampleAmount = sum(PosAmountEachClass) + sum(NegAmountEachClass);
PosAmount = sum(PosAmountEachClass);
NegAmount = sum(NegAmountEachClass);
% length(IntraNeighboringPairs)
% length(InterNeighboringPairs)
if isempty(IntraNeighboringPairs)
    IntraNeighboringPairs = ...
        zeros(1,sum(PosAmountEachClass));    
end
if isempty(InterNeighboringPairs)
    InterNeighboringPairs = ...
        zeros(1,sum(NegAmountEachClass));
end

CumPosAmountEachClass = [0 cumsum(PosAmountEachClass)];
CumNegAmountEachClass = PosAmount + [0 cumsum(NegAmountEachClass)];

if sum(InterNeighboringPairs) == 0 % fast sprase initialize, only work for default setting
    CumAmountEachClass = [0 cumsum(NegAmountEachClass.*PosAmountEachClass)];
    P(:,2) = 1:sum(PosAmountEachClass.*NegAmountEachClass);
    D(:,2) = 1:sum(PosAmountEachClass.*NegAmountEachClass);
    for ip = 1:numel(CumAmountEachClass)-1
        P(CumAmountEachClass(ip)+1:CumAmountEachClass(ip+1),1) = ...
            kron(CumPosAmountEachClass(ip)+1:CumPosAmountEachClass(ip+1),...
            ones(1,NegAmountEachClass(ip)));
        D(CumAmountEachClass(ip)+1:CumAmountEachClass(ip+1),1) = ...
            repmat(CumNegAmountEachClass(ip)+1:CumNegAmountEachClass(ip+1),...
                    1,PosAmountEachClass(ip));
    end
    I = [ones(size(P,1),1); -1*ones(size(D,1),1)];
    PD = [P;D];
    DiffCoefficientSetC = sparse(PD(:,1),PD(:,2),I);
else

% initialization
TempArray = PosAmountEachClass .* NegAmountEachClass;
% sum(TempArray)
% pause
DiffCoefficientSetC = sparse(SampleAmount,sum(TempArray));


% processing
pair_id = 0;
for class_id = 1 : ClassAmount
    for pos_id = 1 : PosAmountEachClass(class_id)
        current_pos_id = CumPosAmountEachClass(class_id) + pos_id;
        current_weight = weight * IntraNeighboringPairs(current_pos_id);
        for neg_id = 1 : NegAmountEachClass(class_id)            
            current_neg_id = CumNegAmountEachClass(class_id) + neg_id;
            current_2_weight = ...
                current_weight * InterNeighboringPairs(current_neg_id - PosAmount);
            pair_id = pair_id + 1;
            DiffCoefficientSetC(current_pos_id,pair_id) = 1+current_2_weight;
            DiffCoefficientSetC(current_neg_id,pair_id) = -(1+current_2_weight);
        end
    end
end
end