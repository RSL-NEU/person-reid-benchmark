function result = evaluateDis(dis, idx_gallery, idx_probe, probeID, galleryID, ...
                                            testCamID, mopts, dopts, ropts)
% Evaluate computed distance with ground truth
% default: sorted by ascend
% Written by Mengran Gou @ 2017
if(strcmp(ropts.rankType,'isr'))
    tmpRank=dis';
else
    if(strcmp(mopts.method,'ranksvm'))
        [disSort,idxSort] = sort(dis,2,'descend');
    else
        [disSort,idxSort] = sort(dis,2,'ascend');
    end
    if(strcmp(dopts.evalType,'clustering') || strcmp(dopts.evalType,'all'))
        galleryID=unique(galleryID,'stable');
        probeID=unique(probeID,'stable');
    end
    IDsort = galleryID(idxSort);
    tmpRank = bsxfun(@eq, IDsort, probeID');
    result.ResMatch = tmpRank;
    if strcmp(dopts.name,'market')
        firstOcc = [];
        AP = [];
        galleryCamID = testCamID(idx_gallery);
        sortCamID = galleryCamID(idxSort);
        probeCamID = testCamID(idx_probe);
        for p = 1:size(tmpRank,1)
            tmpR = tmpRank(p,:);
            junk = sortCamID(p,:)==probeCamID(p) & ...
                IDsort(p,:)==probeID(p); % remove within camera match
            tmpR(junk)=[];            
            firstOcc(p) = min(find(tmpR));
            AP(p) = compute_AP(find(tmpR),1:numel(tmpR));
        end
        tmpRank = hist(firstOcc,1:numel(galleryID));
        result.mAP = mean(AP);
    elseif  strcmp(dopts.name,'airport') || strcmp(dopts.name,'DukeMTMC')
        firstOcc = [];
        AP = [];
        for p = 1:size(tmpRank,1)
            tmpR = tmpRank(p,:);
            firstOcc(p) = min(find(tmpR));
            AP(p) = compute_AP(find(tmpR),1:numel(tmpR));
        end
        tmpRank = hist(firstOcc,1:10000);
        result.mAP = mean(AP);
        result.szGal = numel(galleryID);
        result.szProb = numel(probeID);
    elseif strcmp(dopts.name,'hda')
        firstOcc = [];
        AP = [];
        for p = 1:size(tmpRank,1)
            tmpR = tmpRank(p,:);
            firstOcc(p) = min(find(tmpR));
            AP(p) = compute_AP(find(tmpR),1:numel(tmpR));
        end
        tmpRank = hist(firstOcc,1:numel(galleryID));
        result.mAP = mean(AP);
    else
        tmpRank = sum(tmpRank,1);
    end
    result.Rank = cumsum(tmpRank)./sum(idx_probe);
end