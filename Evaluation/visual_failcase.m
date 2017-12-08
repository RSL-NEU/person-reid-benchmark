clc
% clear 
%%
single = 1;
dataset = 'DukeMTMC';
savefolder = 'Duke'; %'hardeg'
if single
    feat = 'gog';
    algo = 'klfda_exp';
else
    feat = 'gog';
    algo = 'clustering_kissme_srid';
end
% num_w = 30; % worst ones in each partition
num_show = 5; % num of most common ones to show in the end

if strcmp(dataset, 'DukeMTMC')
    campair = [8,7];
end

% load result
resfolder = '../Results';
info = dir(sprintf('%s/rank_%s_%s_%s_*2017*.mat',resfolder,dataset,algo,feat));
if numel(info) > 1
    disp('Non unique results!');
end
load(fullfile(resfolder,info(1).name));
% load ID
load(sprintf('../FeatureExtraction/feature_%s_%s_%dpatch.mat',dataset,feat,6),'personID','camID');
% load partition
id_worst = [];
id_worst_gal = [];
rank_worst = [];
load(sprintf('../TrainTestSplits/Partition_%s.mat',dataset));
for s = 1:numel(partition)
    idx_test = partition(s).idx_test;
    idx_probe = partition(s).idx_probe;
    idx_gallery = partition(s).idx_gallery;
    ids = 1:numel(idx_test);
    id_test = ids(idx_test);
    gID_test = personID(idx_test);
    for pr = 1:size(idx_probe,1)
        if strcmp(dataset, 'DukeMTMC')
            if ~all(metric.cam_pair(pr,:) == campair)
                continue;
            end
        end
        id_probe = id_test(idx_probe(pr,:));
        id_gal = id_test(idx_gallery(pr,:));
        res_match = metric.Res_match{s,pr};
        gID_probe = gID_test(idx_probe(pr,:));
        gID_gallery = gID_test(idx_gallery(pr,:));
        if max(sum(res_match,2)) > 1 || strcmp(dataset,'airport')
            tmp = [];
            for r = 1:size(res_match,1)
                tmp(r) = min(find(res_match(r,:)));
            end
            [rank,sort_probe] = sort(tmp);
            sort_gID_probe = gID_probe(sort_probe);
        else
            [sort_probe,rank] = find(res_match);
            sort_probe = sort_probe';
            rank = rank';
            sort_gID_probe = gID_probe(sort_probe);
            tmpidx = cellfun(@(X,Y) find(X==Y),repmat({gID_gallery},numel(sort_gID_probe),1),...
            mat2cell(sort_gID_probe',ones(1,numel(sort_gID_probe))),'uni',0);
            tmpidx = cell2mat(tmpidx);
            sort_gal = id_gal(tmpidx);
            id_worst_gal = [id_worst_gal, sort_gal];
%             id_worst_gal = [id_worst_gal,sort_gal(end:-1:end-min(numel(sort_probe),num_w)+1)];
        end
        
        sort_probe = id_probe(sort_probe);
        id_worst = [id_worst, sort_probe];
        rank_worst = [rank_worst, rank];
%         id_worst = [id_worst, sort_probe(end:-1:end-min(numel(sort_probe),num_w)+1)];
%         rank_worst = [rank_worst, rank(end:-1:end-min(numel(sort_probe),num_w)+1)];
    end
end
[~,idx_sort] = sort(rank_worst,'descend');
id_worst = id_worst(idx_sort);
rank_worst = rank_worst(idx_sort);
if ~isempty(id_worst_gal)
    id_worst_gal = id_worst_gal(idx_sort);
end
%% load IDs
if ~single
    personID = unique(personID,'stable');
    personID = repmat(personID,1,2);
    camID = [ones(1,numel(personID)/2),ones(1,numel(personID)/2)*2];
end
pID_worsts = personID(id_worst(:));
rank_worsts = rank_worst(:);
galId_worsts = id_worst_gal(:);

pID_worst = [];
galId_worst = [];
probeID = [];
rank_worst = [];
for i = 1:numel(pID_worsts) % starting from the worst
    tmpID = pID_worsts(i);
    if ismember(tmpID,pID_worst) % already exist
        continue;
    end
    probeID = [probeID,id_worst(i)];
    pID_worst = [pID_worst,tmpID];
    rank_worst = [rank_worst,rank_worsts(i)];
    if ~isempty(id_worst_gal)
        galId_worst = [galId_worst,galId_worsts(i)];
    end
end

%% load images
f_show = 5; %maximum frame to show in each camera
if single
%     load(sprintf('../FeatureExtraction/img_%s.mat',dataset));
else
    switch dataset
        case 'caviar'
            imfolder = '../../data/caviar/';
        case 'raid_12'
            imfolder = '../../data/RAiD/12/';
        case 'raid_13'
            imfolder = '../../data/RAiD/13/';
        case 'raid_14'
            imfolder = '../../data/RAiD/14/';
        case 'ward_12'
            imfolder = '../../data/WARD/12/';
        case 'ward_13'
            imfolder = '../../data/WARD/13/';
        case 'saivt_38'
            imfolder = '../../data/saivt/38/';
        case 'saivt_58'
            imfolder = '../../data/saivt/58/';
        case 'ilidsvid'
            imfolder = '../../data/i-LIDS-VID/multi_shot/';
        case 'prid'
            imfolder = '../../data/prid_2011/multi_shot_DVR/';
    end    
end
for p = 1:min(numel(pID_worst),num_show)
    if single
        im_probe = imgs{probeID(p)};
    else
        cam_p = camID(probeID(p));
        
        tmpID = pID_worst(p);
        if strcmp(dataset,'caviar')
            folder_p = fullfile(imfolder,sprintf('cam%d/person%d',cam_p,tmpID));
            info = dir(fullfile(folder_p,'*.*'));
        elseif strcmp(dataset,'prid')
            folder_p = fullfile(imfolder,sprintf('cam%d/person%03d',cam_p,tmpID));
            info = dir(fullfile(folder_p,'*.*'));
        else
            folder_p = fullfile(imfolder,sprintf('cam%d/%04d',cam_p,tmpID));
            info = dir(fullfile(folder_p,'*.*'));
        end
        info = info(3:end);
        if strcmp(info(1).name(end-1:end),'db')
            info = info(2:end);
        end
    end
    disp(sprintf('The most difficult case for %s is ID-%d with mean rank %d',dataset,pID_worst(p),...
        rank_worst(p)));
    if single
        imwrite(imresize(im_probe,[128,48]),sprintf('%s/%s_r%d_p%d_c%d_Probe.png',savefolder,dataset,rank_worst(p),...
            pID_worst(p),camID(probeID(p))));
    else
        for i = 1:floor(numel(info)/f_show):numel(info)
            tmpim = imread(fullfile(folder_p,info(i).name));
            imwrite(imresize(tmpim,[128 48]),sprintf('%s/%s_p%d_c%d_Probe_f%d_mr%d.png',savefolder,dataset,pID_worst(p),...
                camID(probeID(p)),i,rank_worst(p)));
        end
    end
    if ~isempty(galId_worst)
        idx_gal = 1:numel(personID) == galId_worst(p);
    else
        if numel(unique(camID))<2
            idx_gal = ismember(personID,pID_worst(p));
            idx_gal(probeID(p)) = 0;
        else
            idx_gal = ismember(personID,pID_worst(p)) & camID~=camID(probeID(p));
        end
    end
    camID_gal = camID(idx_gal);
    ucamID = unique(camID_gal);
    if single
        for u = 1:numel(ucamID)
            tmpims = imgs(idx_gal & camID==ucamID(u));
            tf_show = min(f_show,numel(tmpims));
            tmpim = tmpims(randsample(numel(tmpims),tf_show));
            for f = 1:tf_show
                imwrite(imresize(tmpim{f},[128,48]),sprintf('%s/%s_r%d_p%d_c%d_f%d.png',savefolder,dataset,rank_worst(p),...
                    pID_worst(p),ucamID(u),f));
            end
        end
    else
        if strcmp(dataset,'caviar')
            folder_g = fullfile(imfolder,sprintf('cam%d/person%d',camID_gal,tmpID));
            info = dir(fullfile(folder_g,'*.*'));
        elseif strcmp(dataset,'prid')
            folder_g = fullfile(imfolder,sprintf('cam%d/person%03d',camID_gal,tmpID));
            info = dir(fullfile(folder_g,'*.*'));
        else
            folder_g = fullfile(imfolder,sprintf('cam%d/%04d',camID_gal,tmpID));
            info = dir(fullfile(folder_g,'*.*'));
        end
        info = info(3:end);
        if strcmp(info(1).name(end-1:end),'db')
            info = info(2:end);
        end
        for i = 1:floor(numel(info)/f_show):numel(info)
            tmpim = imread(fullfile(folder_g,info(i).name));
            imwrite(imresize(tmpim,[128 48]),sprintf('%s/%s_p%d_c%d_f%d_mr%d.png',savefolder,dataset,pID_worst(p),...
                camID_gal,i,rank_worst(p)));
        end
    end
end
    
    
