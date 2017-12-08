clc
clear 
%%
resfolder = '../Results';
infos = dir([resfolder  '/rank_DukeMTMC_*.mat']);
for i = 1:numel(infos)
    i
    load(fullfile(resfolder,infos(i).name));
    metric_ori = metric;
    
    idp = metric_ori.cam_pair(:,2) > 0;
    % keep pair wise
    metric.mAP = metric_ori.mAP(idp);
    if ~isempty(metric.Res_match)
        metric.Res_match = metric_ori.Res_match(idp);
    end
    metric.szGal = metric_ori.szGal(idp);
    metric.szProb = metric_ori.szProb(idp);
    metric.rank = metric_ori.rank(idp,:);
    metric.cam_pair = metric_ori.cam_pair(idp,:);
    
    rank = sum(bsxfun(@times,metric.rank,metric.szProb),1)./sum(metric.szProb);
    mAP = sum(metric.mAP.*metric.szProb)/sum(metric.szProb);
    
    savename = ['rank_DukeReIDpair_' infos(i).name(15:end)];
    save(fullfile(resfolder,savename),'metric','rank','mAP');
    
    % keep mixed gallery
    metric.mAP = metric_ori.mAP(~idp);
    if ~isempty(metric.Res_match)
        metric.Res_match = metric_ori.Res_match(~idp);
    end
    metric.szGal = metric_ori.szGal(~idp);
    metric.szProb = metric_ori.szProb(~idp);
    metric.rank = metric_ori.rank(~idp,:);
    metric.cam_pair = metric_ori.cam_pair(~idp,:);
    
    rank = sum(bsxfun(@times,metric.rank,metric.szProb),1)./sum(metric.szProb);
    mAP = sum(metric.mAP.*metric.szProb)/sum(metric.szProb);
    
    savename = ['rank_DukeReIDmix_' infos(i).name(15:end)];
    save(fullfile(resfolder,savename),'metric','rank','mAP');
    
end