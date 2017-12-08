function [PosDiffSetC,NegDiffSetC,PosAmountEachClass,NegAmountEachClass] = ...
    MakePosNegSubset(DataSetC,CountOfSampleEachClass,IsAbs,GroupPointAmount)

% get basic information
ClassCount = length(CountOfSampleEachClass);
SampleCount = sum(CountOfSampleEachClass);
Dim_NewData = round(size(DataSetC,1) / GroupPointAmount);
Dim_OriData = size(DataSetC,1);

% get the difference images
IntraDiffCount = sum(CountOfSampleEachClass .* (CountOfSampleEachClass - 1)) / 2;
InterDiffCount = sum(CountOfSampleEachClass .* (SampleCount - CountOfSampleEachClass));
PosDiffSetC = zeros(Dim_NewData,IntraDiffCount);
NegDiffSetC = zeros(Dim_NewData,InterDiffCount);
PosAmountEachClass = zeros(1,IntraDiffCount);
NegAmountEachClass = zeros(1,IntraDiffCount);

startpoint_pos = 0;
startpoint_neg = 0;
endpoint_pos = 0;
endpoint_neg = 0;
startpoint = 0;
endpoint = 0;
intra_pair_id = 0;
for class_id = 1 : ClassCount
    startpoint = endpoint + 1;
    endpoint = endpoint + CountOfSampleEachClass(class_id);
    CurrentClassDataSetC = DataSetC(:,startpoint : endpoint);
    RestClassDataSetC = DataSetC;
    RestClassDataSetC(:,startpoint : endpoint) = [];
    RestClassSampleAmount = size(RestClassDataSetC,2);
    
    for sample_id = 1 : (CountOfSampleEachClass(class_id) - 1)
        intra_pair_id = intra_pair_id + 1;
        rest_inner_class = CountOfSampleEachClass(class_id) - sample_id;
        startpoint_pos = endpoint_pos + 1;
        endpoint_pos = endpoint_pos + rest_inner_class;
        startpoint_neg = endpoint_neg + 1;
        endpoint_neg = endpoint_neg + RestClassSampleAmount;        
        current_vector = CurrentClassDataSetC(:,sample_id);
        
        % get intra class diff vectors        
        DiffDataSetC = ...
            current_vector * ones(1,rest_inner_class) ...
            - CurrentClassDataSetC(:,(sample_id + 1) : end);
        if IsAbs == 1
            DiffDataSetC = abs(DiffDataSetC);
        end
        if GroupPointAmount > 1
            startpoint_group = 0;
            endpoint_group = 0;
            sub_dim = 0;
            
            while 1
                startpoint_group = endpoint_group + 1;
                endpoint_group = endpoint_group + GroupPointAmount;
                if endpoint_group > Dim_OriData
                    endpoint_group = Dim_OriData;
                end
                sub_dim = sub_dim + 1;
                PosDiffSetC(sub_dim,startpoint_pos : endpoint_pos) = ...
                    sum(DiffDataSetC(startpoint_group : endpoint_group,:));
                if endpoint_group == Dim_OriData
                    break;
                end
            end
        else
            PosDiffSetC(:,startpoint_pos : endpoint_pos) = DiffDataSetC;
        end        
        PosAmountEachClass(intra_pair_id) = size(DiffDataSetC,2);
        
        % get inter class diff vectors
        DiffDataSetC = ...
            current_vector * ones(1,RestClassSampleAmount) ...
            - RestClassDataSetC;
        if IsAbs == 1
            DiffDataSetC = abs(DiffDataSetC);
        end
        if GroupPointAmount > 1
            startpoint_group = 0;
            endpoint_group = 0;
            sub_dim = 0;
            
            while 1
                startpoint_group = endpoint_group + 1;
                endpoint_group = endpoint_group + GroupPointAmount;
                if endpoint_group > Dim_OriData
                    endpoint_group = Dim_OriData;
                end
                sub_dim = sub_dim + 1;
                NegDiffSetC(sub_dim,startpoint_neg : endpoint_neg) = ...
                    sum(DiffDataSetC(startpoint_group : endpoint_group,:));
                if endpoint_group == Dim_OriData
                    break;
                end
            end
        else
            NegDiffSetC(:,startpoint_neg : endpoint_neg) = DiffDataSetC;
        end
        NegAmountEachClass(intra_pair_id) = size(DiffDataSetC,2);
    end
end