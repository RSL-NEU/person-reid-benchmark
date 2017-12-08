function [X,pID,cID]=getFeaturesSplits_MultiShot(features,personID,camID,ind)

uCamID=unique(camID);
cam1_features=features(find(camID==uCamID(1)),:);
cam1_id=personID(find(camID==uCamID(1)));
cam2_features=features(find(camID==uCamID(2)),:);
cam2_id=personID(find(camID==uCamID(2)));

X=[];pID=[];cID=[];currFeat_cam1=[];currFeat_cam2=[];currId_cam1=[];currId_cam2=[];currCam1Id=[];currCam2Id=[];
tmp=find(ind(1:length(ind)/2)==1);
upID=unique(personID);
for i=1:length(tmp)
    currID=upID(tmp(i));
    
    tmpFeat=cam1_features(find(cam1_id==currID),:);
    currFeat_cam1=[currFeat_cam1;tmpFeat];
    currId_cam1=[currId_cam1 currID*ones(1,size(tmpFeat,1))];
    currCam1Id=[currCam1Id ones(1,size(tmpFeat,1))];
    
    tmpFeat=cam2_features(find(cam2_id==currID),:);
    currFeat_cam2=[currFeat_cam2;tmpFeat];    
    currId_cam2=[currId_cam2 currID*ones(1,size(tmpFeat,1))];  
    currCam2Id=[currCam2Id 2*ones(1,size(tmpFeat,1))];
end

X=[currFeat_cam1;currFeat_cam2];
pID=[currId_cam1 currId_cam2];
cID=[currCam1Id currCam2Id];