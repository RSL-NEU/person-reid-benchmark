function opts = setParam(paramType,paramSet)
% set parameters and fill up the default values

switch paramType
    case 'dataset'
        opts = struct('name','viper',...       % dataset name
                      'datafolder','./Data',...
                      'evalType',[],...
                      'pair',[]);              % specific camera pairs 
        opts = vl_argparse(opts,paramSet);
        % take care pair
        if ismember(opts.name,{'saivt','raid','ward'}) 
            if isempty(opts.pair)           
                switch opts.name
                    case 'saivt'
                        pairs='38';
                    case 'raid'
                        pairs='12';
                    case 'ward'
                        pairs='12';
                end
                opts = vl_argparse(opts,'pair',pairs);
            end
        else
            if ~isempty(opts.pair)
                opts.pair = [];
            end
        end        
        opts.imgSize = get_const('imgSize',opts);
        % take care multishot datasets
        if get_const('isMulti',opts)
            opts.evalType = 'featureAverage';
            if(strcmp(opts.evalType,'clustering'))
                opts.numClusters=10;
            end
        end        
    case 'feature'
        opts = struct('featureType','whos',... % feature type
                      'numRow',6,...           % number of splitted rows    
                      'numCol',1,...           % number of splitted cols
                      'overlap',0,...          % indicator for overlapped split
                      'patchSize',[],...       % patch size
                      'stepSize',[],...        % step size 
                      'doPCA',0,...            % indicator for PCA dimensionality reduction
                      'pcadim',100);           % PCA dimensions
        opts = vl_argparse(opts,paramSet);
    case 'metric'
        opts = struct('method','xqda',...      % metric learning method
                      'kernels',[],...         % kernel types                      
                      'd',40,...               % projection dimensions 
                      'npratio',10,...         % negtive-positive pair ratio; 0 for all negtive pairs (TIME CONSUMING)
                      'name','metricLearning');% 'l2' for no metric learning
        opts = vl_argparse(opts,paramSet);
    case 'ranking'
        opts = struct('rankType','rnp',...     % rank type for multi-shot
                      'saveMetric',1,...       % indicator for saving learned metric
                      'saveInterm',1,...       % indicator for saving pair-wise scores
                      'param',[]);         
        opts = vl_argparse(opts,paramSet);

end

