function [DiffSetC] = ...
    MakeDiffSubset(ProtypeDataSetC,ProbeVector,IsAbs,GroupPointAmount)

% get basic information
ProtypeSampleCount = size(ProtypeDataSetC,2);
Dim_NewData = round(size(ProtypeDataSetC,1) / GroupPointAmount);
Dim_OriData = size(ProtypeDataSetC,1);

% get the difference images
DiffSetC = zeros(Dim_NewData,ProtypeSampleCount);
% TempDiffDataSetC = ...
%     ProbeVector * ones(1,ProtypeSampleCount) - ProtypeDataSetC;
TempDiffDataSetC = bsxfun(@minus,ProtypeDataSetC,ProbeVector);
if IsAbs == 1
    TempDiffDataSetC = abs(TempDiffDataSetC);
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
        DiffSetC(sub_dim,:) = ...
            sum(TempDiffDataSetC(startpoint_group : endpoint_group,:));
        if endpoint_group == Dim_OriData
            break;
        end
    end
else
    DiffSetC = TempDiffDataSetC;
end