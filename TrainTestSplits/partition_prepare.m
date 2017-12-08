clc
clear all
data.name = 'v47';
data.pair = [];

if(isempty(data.pair))
    load(['../FeatureExtraction/feature_' data.name '_sdc_6patch.mat'],'personID','camID');
    load(['Split_' data.name '.mat']);
else
    load(['../FeatureExtraction/feature_' data.name '_' data.pair '_sdc_6patch.mat'],'personID','camID');
    load(['Split_' data.name '_' data.pair '.mat']);
end

for s = 1:size(split)
    %     switch data.name
    %         case 'gird'
    [idx_train,idx_test,idx_probe,idx_gallery] = getTrainTestSplit(personID,camID,data,s);
    if strcmp(data.name,'prid') || strcmp(data.name,'ilidsvid') || ...
            strcmp(data.name,'saivt') || strcmp(data.name,'ward') || strcmp(data.name,'raid') || ...
            strcmp(data.name,'caviar') || strcmp(data.name,'v47')
        %tmp_breakP = diff([-1 personID],1,2); % average feature setting
        %uni_idx = any(tmp_breakP,1);
        %trainID = personID(uni_idx);
        %trainCamID = camID(uni_idx);
        personID=unique(personID,'stable');
        personID=[personID personID];
        switch data.name
            case 'caviar'
                camID=[ones(1,50) 2*ones(1,50)];
            case 'ilidsvid'                
                camID=[ones(1,300) 2*ones(1,300)];
            case 'v47'
                camID=[ones(1,47) 2*ones(1,47)];
            case 'prid'
                camID=[ones(1,178) 2*ones(1,178)];
            case 'saivt'
                switch data.pair
                    case '38'
                        camID=[ones(1,99) 2*ones(1,99)];
                    case '58'
                        camID=[ones(1,103) 2*ones(1,103)];
                end
            case 'raid'
                switch data.pair
                    case '12'
                        camID=[ones(1,43) 2*ones(1,43)];
                    case '13'
                        camID=[ones(1,42) 2*ones(1,42)];
                    case '14'
                        camID=[ones(1,42) 2*ones(1,42)];                        
                end
            case 'ward'
                switch data.pair
                    case '12'
                        camID=[ones(1,70) 2*ones(1,70)];
                    case '13'
                        camID=[ones(1,70) 2*ones(1,70)];
                end
        end
       trainID=personID(idx_train);
       trainCamID=camID(idx_train);
    elseif strfind(data.name,'cuhk') 
        trainID = personID(idx_train);
        trainCamID = camID(idx_train);
        trainCamID = mod(trainCamID,10);
    else
        trainID = personID(idx_train);
        trainCamID = camID(idx_train);
    end
    if strcmp(data.name,'3dpes')
        [ix_pos_pair, ix_neg_pair]=GeneratePair(trainID);
    else
        [ix_pos_pair, ix_neg_pair]=GeneratePair(trainID,trainCamID,0);
    end
    ix_pos_pair = uint16(ix_pos_pair);
    ix_neg_pair = uint16(ix_neg_pair);
    partition(s).idx_train = idx_train;
    partition(s).idx_test = idx_test;
    partition(s).idx_probe = idx_probe;
    partition(s).idx_gallery = idx_gallery;
    partition(s).ix_pos_pair = ix_pos_pair;
    partition(s).ix_neg_pair = ix_neg_pair;
    %         case 'viper'
    %             [idx_train,idx_test,idx_probe,idx_gallery] = getTrainTestSplit(personID,camID,data,s);
%             trainID = personID(idx_train);
%             trainCamID = camID(idx_train);    
%             [ix_pos_pair, ix_neg_pair]=GeneratePair(trainID,trainCamID,0);
%             ix_pos_pair = uint16(ix_pos_pair);
%             ix_neg_pair = uint16(ix_neg_pair);
%             partition(s).idx_train = idx_train;
%             partition(s).idx_test = idx_test;
%             partition(s).idx_probe = idx_probe;
%             partition(s).idx_gallery = idx_gallery;
%             partition(s).ix_pos_pair = ix_pos_pair;
%             partition(s).ix_neg_pair = ix_neg_pair;
end
if(isempty(data.pair))
    save(['Partition_' data.name '.mat'],'partition','-v7.3');
else
    save(['Partition_' data.name '_' data.pair '.mat'],'partition','-v7.3');
end
    