%% Main script to generate results
rng('default'); % fix random seed

%% initialize variables
rank=[];%zeros(tsize,1);
mAP = [];
szGal = [];
szProb = [];
time_train = [];
ResMatch = {};

% number of splits
num_split = 10;
if strfind(dopts.name, 'cuhk_')
    if(strcmp(fopts.featureType,'IDE_ResNet') || strcmp(fopts.featureType,'IDE_CaffeNet') ...
            || strcmp(fopts.featureType,'IDE_VGGNet'))
        num_split=10;
    elseif(strcmp(mopts.method,'NFST'))
        num_split=1;
    else
        num_split = 20;
    end
end
if strcmp(dopts.name, 'market') || strcmp(dopts.name, 'airport') ... 
    || strcmp(dopts.name, 'hda') ||  strcmp(dopts.name, 'DukeMTMC') || ...
    strcmp(dopts.name, 'mtmc17')
    num_split = 1;
end
%% Feature extraction/loading
% compute patch size or number of patches
if isempty(fopts.patchSize)
    fopts.patchSize = zeros(1,2);
    fopts.stepSize = zeros(1,2);
    if fopts.overlap
        fopts.patchSize(1) = dopts.imgSize(1)/(fopts.numRow+1) * 2;
        fopts.patchSize(2) = dopts.imgSize(2)/(fopts.numCol+1) * 2;
        fopts.stepSize = fopts.patchSize/2;
    else
        fopts.patchSize(1) = dopts.imgSize(1)/(fopts.numRow);
        fopts.patchSize(2) = dopts.imgSize(2)/(fopts.numCol);
        fopts.stepSize = fopts.patchSize;
    end
end
% fopts.patchSize = floor(fopts.patchSize);
% fopts.stepSize = floor(fopts.stepSize);
fopts.numPatch = floor(((dopts.imgSize(1) - fopts.patchSize(1))/fopts.stepSize(1)+1))...
                    * floor(((dopts.imgSize(2) - fopts.patchSize(2))/fopts.stepSize(2)+1));
fopts.featureFile = get_const('file_feature',dopts,fopts);


% load partition
dopts.partitionFile = get_const('file_partition',dopts);
if exist(dopts.partitionFile,'file') ~= 0 
    load(dopts.partitionFile)
else
    error('Please download the partition file first')
end

% load/extract features
if exist(fopts.featureFile,'file') ~= 0
    % load pre-computed feature
    feat_precompute = load(fopts.featureFile);
    features = feat_precompute.features;
    personID = feat_precompute.personID;
    camID = feat_precompute.camID;
    clear feat_precompute
else
    % parse all images 
    [ imgs,camID,personID ] = parsingDataset( dopts,partition );
    % compute feature
    if strcmp(fopts.featureType,'ldfv') % extract ldfv feature based on train/test split       
        for s = 1:numel(partition)
            fopts.idx_train = partition(s).idx_train;
            [features{s},fopts] = ComputeFeatures(imgs,fopts);
        end
    else
        [features,fopts] = ComputeFeatures(imgs,fopts);
    end     
    save(fopts.featureFile, 'features','camID','personID','fopts','-v7.3');
end

% clustering for multi-shot
if (~(strcmp(fopts.featureType,'ldfv')||strcmp(fopts.featureType,'IDE_ResNet')||strcmp(fopts.featureType,'IDE_CaffeNet')...
        ||strcmp(fopts.featureType,'IDE_VGGNet')))
    if ~isempty(dopts.evalType)
        [features,personID,camID]=parseFeaturesMultiShot(features,personID,camID,dopts);
    end
end
oriFeat = features;
oriCamId=camID;
oripersonId=personID;
%% ------------------ metric learning ---------------------
disp('################################################################')
fprintf('Dataset---%s\t [pair:%s]\n',dopts.name,dopts.pair);
fprintf('Feature---%s\t [# Patch:%d\t PCA:%d]\n',fopts.featureType,fopts.numPatch,fopts.doPCA);
fprintf('Metric----%s\t [kernel:%s]\n',mopts.method,mopts.kernels);
disp('################################################################')
disp('Start evaluation...')
fprintf('Accuracy: \tRank1 \tRank5 \tRank10 \tRank20\n')
for s=1:num_split
    % retrieve train/test information
    idx_train = partition(s).idx_train;
    idx_test = partition(s).idx_test;
    idx_probe = partition(s).idx_probe;
    idx_gallery = partition(s).idx_gallery;
    idx_pos_pair = partition(s).ix_pos_pair;
    idx_neg_pair = partition(s).ix_neg_pair; 

    if iscell(oriFeat) % special care for LDFV features
        features = oriFeat{s};
%         if(size(dopts.evalType,1)~=0)
%             [features,personID,camID]=parseFeaturesMultiShot(features,oripersonId,oriCamId,data);
%         end
    else 
        features = oriFeat;
    end
    
        % PCA 
    if fopts.doPCA ~= 0
        disp('PCA applied!');
        [U,mu,vars] = pca(features(idx_train,:)');
        [Yhat,Xhat,avsq] = pcaApply(features',U,mu,fopts.pcadim);
        features = Yhat';
    end
%% --------------------- training ------------------------
    if((strcmp(dopts.evalType,'clustering') || strcmp(dopts.evalType,'all')) && ~strcmp(dopts.name,'DukeMTMC'))
        [X,trainID,trainCamID]=getFeaturesSplits_MultiShot(features,personID,camID,idx_train);                
        % Average features prior to training
        [X,trainID,trainCamID]=parseFeaturesMultiShot(X,trainID,trainCamID,struct('evalType','featureAverage'));
    else
        X = features(idx_train,:);
        trainID = personID(idx_train);
        trainCamID = camID(idx_train);
    end 

    % mean removal + L2 norm for GOG and deep feature
    if(strcmp(fopts.featureType,'gog')||strcmp(fopts.featureType,'IDE_CaffeNet')...
            ||strcmp(fopts.featureType,'IDE_ResNet')||strcmp(fopts.featureType,'IDE_VGGNet'))
        meanX=mean(X,1);
        X = ( X - repmat(meanX, size(X,1), 1));
        X=bsxfun(@times, X, 1./sqrt(sum(X.^2, 2)));
    end

    % metric learning
    if(~strcmp(mopts.name,'l2'))
        metric=learnProjectionMatrix(X,trainCamID,trainID,idx_pos_pair,...
                                        idx_neg_pair,mopts,dopts);
        time_train(s) = metric.options.time_train;
    else
        metric.options=[];
        time_train(s)=0;
    end            
%% --------------------- testing -------------------------    
    % Pre-processing 
    if((strcmp(dopts.evalType,'clustering') || strcmp(dopts.evalType,'all')) && ~strcmp(dopts.name,'DukeMTMC'))
        [testFeat,testID,testCamID]=getFeaturesSplits_MultiShot(features,personID,camID,idx_test);
    else
        testFeat = features(idx_test,:);
        testID = personID(idx_test);
        testCamID = camID(idx_test);            
    end
    % mean removal + L2 for GOG and deep feature
    if(strcmp(fopts.featureType,'gog')||strcmp(fopts.featureType,'IDE_ResNet')...
            ||strcmp(fopts.featureType,'IDE_CaffeNet')||strcmp(fopts.featureType,'IDE_VGGNet'))
        testFeat = ( testFeat - repmat(meanX, size(testFeat,1), 1));
        testFeat=bsxfun(@times, testFeat, 1./sqrt(sum(testFeat.^2, 2)));
    end
    
    % Compute Euclidean distance to determine rank
    % test feature mapping
    [ testFeatProj ] = testProjection( testFeat, X, metric, mopts );
    
    % loop over probe cameras  
    if strcmp(dopts.name,'DukeMTMC')
        pr_s = 57; % Only evaluate fix-one-camera protocol for DukeMTMC
    else
        pr_s = 1;
    end    
    for pr = 1:size(idx_probe,1)
        dis = [];
        if(strcmp(dopts.evalType,'clustering') || strcmp(dopts.evalType,'all')) % multi shot ranking
            if(~strcmp(dopts.name,'DukeMTMC'))
                uCamID=unique(testCamID,'stable');
                probeFeat=testFeatProj(find(testCamID==uCamID(1)),:);
                probeID=testID(find(testCamID==uCamID(1)));
                galleryFeat=testFeatProj(find(testCamID==uCamID(2)),:);
                galleryID=testID(find(testCamID==uCamID(2)));
                [dis,param]=multiShotRanking(probeFeat,probeID,galleryFeat,galleryID,ropts,metric,dopts);
                evalType.method.rankType.param=param;
            else
                probeFeat = testFeatProj(idx_probe(pr,:),:);
                probeID = testID(idx_probe(pr,:));                
                galleryFeat = testFeatProj(idx_gallery(pr,:),:);
                galleryID = testID(idx_gallery(pr,:));
                [dis,param] = multiShotRanking(probeFeat,probeID,galleryFeat,galleryID,evalType.method,metric,data);
                ropts.param=param;
            end                    
        else % single shot distance 
            probeFeat = testFeatProj(idx_probe(pr,:),:);
            probeID = testID(idx_probe(pr,:));                
            galleryFeat = testFeatProj(idx_gallery(pr,:),:);
            galleryID = testID(idx_gallery(pr,:));
            dis = singleShotRanking(probeFeat, galleryFeat,metric,mopts);
        end              

        % evaluate with ground truth
        resultStruct = evaluateDis(dis, idx_gallery(pr,:), idx_probe(pr,:), probeID, galleryID, testCamID, mopts, dopts, ropts);        
        
        % keep results
        rank = [rank; resultStruct.Rank];
        ResMatch{s,pr} = resultStruct.ResMatch;
        if isfield(resultStruct,'mAP')          
            mAP = [mAP, resultStruct.mAP];
        end
        if isfield(resultStruct,'szGal')
            szGal = [szGal, resultStruct.szGal];
            szProb = [szProb, resultStruct.szProb];
        end        
        fprintf('s%02d--p%02d: \t%.2f \t%.2f \t%.2f \t%.2f\n',s,pr,resultStruct.Rank(1)*100,...
            resultStruct.Rank(5)*100,resultStruct.Rank(10)*100,resultStruct.Rank(20)*100)
    end            
    
end

time_train = mean(time_train);
metric.time_train = time_train;
if ropts.saveInterm    
    metric.ResMatch = ResMatch;
end
% Average rank
if strcmp(dopts.name,'airport') % average rank for each camera pair
    metric.mAP = mAP*100;
    metric.szGal = szGal;
    metric.rank = rank;
    metric.galCam = partition.galCam;
    rank = mean(rank,1)*100;
elseif strcmp(dopts.name,'DukeMTMC')
    metric.mAP = mAP*100;
    metric.szGal = szGal;
    metric.szProb = szProb;
    metric.rank = rank;
    metric.cam_pair = partition.cam_pairs;
    rank = mean(rank,1)*100;
elseif strcmp(dopts.name,'market') 
    metric.mAP = mAP*100;
    rank = mean(rank,1)*100;
else
    rank = mean(rank,1)*100;
end
fprintf('----------------------------------------------\n')
fprintf('Average: \t%.2f \t%.2f \t%.2f \t%.2f\n',rank(1), rank(5), rank(10), rank(20));

%% Save results
systime = clock;
time = sprintf('%04d_%02d_%02d_%02d_%02d_%2.f',systime(1),systime(2),systime(3),systime(4),systime(5),systime(6));
savefile = get_const('file_result',dopts,fopts,mopts,ropts);
savefile = strcat(savefile,'_',time,'.mat');
save(savefile,'rank','metric','dopts','fopts','mopts','ropts');


    
    