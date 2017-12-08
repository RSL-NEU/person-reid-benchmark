function output = get_const(const,dopts,fopts,mopts,ropts)
% output = get_const(const,dopts,fopts,mopts,ropts)
% obtain constant variables

datasetMulti = {'ilidsvid','prid','saivt','raid','ward','caviar'};
datasetSingle = {'viper','grid','3dpes','hda','market','airport','DukeMTMC',...
                  'cuhk_detected','cuhk01','cuhk02'};
              
switch const
    case 'isMulti'
        output = ismember(dopts.name,datasetMulti);
    case 'imgSize'
        if ismember(dopts.name,{'viper','cuhk01','caviar','3dpes','grid'})
            output = [128 48];
        else
            output = [128 64];
        end            
    case 'file_feature'
        output = sprintf('./FeatureExtraction/feature_%s_%s_%dpatch.mat',...
                    dopts.name,fopts.featureType,fopts.numPatch);
    case 'file_partition'
        if isempty(dopts.pair)
            output = sprintf('./TrainTestSplits/Partition_%s.mat', dopts.name);
            if strcmp(dopts.name,'DukeMTMC')
                output = sprintf('./TrainTestSplits/Partition_%s.mat', 'DukeReID');
            end
        else
            output = sprintf('./TrainTestSplits/Partition_%s_%s.mat', dopts.name,dopts.pair);
        end
    case 'file_result'
        output = sprintf('./Results/rank_%s',dopts.name);
        if ismember(dopts.name,datasetMulti) % multishot
            if ~isempty(dopts.pair)
                output = strcat(output,'_', dopts.pair);
            end
            output = strcat(output, '_', dopts.evalType);
            output = strcat(output, '_', mopts.method);
            output = strcat(output, '_', ropts.rankType);
        else % single shot
            output = strcat(output, '_', mopts.method);
            if ~isempty(mopts.kernels)
                output = strcat(output, '_', mopts.kernels);
            end
        end
        output = strcat(output,'_',fopts.featureType,'_s',num2str(fopts.numPatch));
        if fopts.doPCA
            output = strcat(output,'_pca',num2str(fopts.pcadim));
        end        
end