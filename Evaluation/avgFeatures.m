function [f,upids,ucids] = avgFeatures(features,personID,camID)

uCamID=unique(camID);
cam1_features=features(find(camID==uCamID(1)),:);
cam1_id=personID(find(camID==uCamID(1)));
cam2_features=features(find(camID==uCamID(2)),:);
cam2_id=personID(find(camID==uCamID(2)));

[cam1_features,cam1_id]=avgFeatures1(cam1_features,cam1_id);
[cam2_features,cam2_id]=avgFeatures1(cam2_features,cam2_id);
f=[cam1_features;cam2_features];
upids=[cam1_id cam2_id];
ucids=[ones(1,length(cam1_id)) 2*ones(1,length(cam2_id))];

function [f,uids] = avgFeatures1(feat,ids)

uids=unique(ids,'stable');
f=[];
for i=1:length(uids)
    currId=uids(i);
    ids1=find(ids==currId);
    tmp=zeros(1,size(feat,2));
    for j=1:length(ids1)
        tmp=tmp+feat(ids1(j),:);
    end
    tmp=tmp./length(ids1);
    f=[f;tmp];
end