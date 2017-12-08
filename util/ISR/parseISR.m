function [galleryFeatures,galleryId]=parseISR(galleryFeatures,galleryId)

gf=[];
gid=[];

tmp=unique(galleryId);
for i=1:length(tmp)
    curr=tmp(i);
    idx=find(galleryId==curr);
    if(length(idx)>5)
        idx1=randperm(length(idx),5);
        idx=idx(idx1);
    end
    gf=[gf galleryFeatures(:,idx)];
    gid=[gid curr*ones(1,length(idx))];
end
galleryFeatures=gf;
galleryId=gid;