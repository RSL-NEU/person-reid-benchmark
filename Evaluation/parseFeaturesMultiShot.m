
function [f,upids,ucids] = parseFeaturesMultiShot(features,personID,camID,data)


uCamID=unique(camID,'stable');
cam1_features=features(find(camID==uCamID(1)),:);
cam1_id=personID(find(camID==uCamID(1)));
cam2_features=features(find(camID==uCamID(2)),:);
cam2_id=personID(find(camID==uCamID(2)));

switch data.evalType
    case 'featureAverage'     
        [cam1_features,cam1_id]=avgFeatures1(cam1_features,cam1_id);
        [cam2_features,cam2_id]=avgFeatures1(cam2_features,cam2_id);        
    case 'clustering'
        [cam1_features,cam1_id]=clusterFeatures1(cam1_features,cam1_id,data.numClusters);
        [cam2_features,cam2_id]=clusterFeatures1(cam2_features,cam2_id,data.numClusters);
end
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
end


function [f,nids]=clusterFeatures1(feat,ids,nClusters)

uids=unique(ids,'stable');
f=[];nids=[];
for i=1:length(uids)
    currId=uids(i);
    ids1=find(ids==currId);
    currFeat=feat(ids1,:);
    % Cluster features and retrieve cluster centers
    if(size(currFeat,1)>nClusters)
        [~,C]=litekmeans(double(currFeat'),nClusters);
        C=C';
    else
        C=currFeat;
    end
    f=[f;C];
    nids=[nids currId*ones(1,size(C,1))];
end
end

end

function [label,C] = litekmeans(X, k)
% Perform k-means clustering.
%   X: d x n data matrix
%   k: number of seeds
% Written by Michael Chen (sth4nth@gmail.com).
n = size(X,2);
last = 0;
label = ceil(k*rand(1,n));  % random initialization
while any(label ~= last)
    [u,~,label] = unique(label);   % remove empty clusters
    label = label(:)';
    k = length(u);
    E = sparse(1:n,label,1,n,k,n);  % transform label into indicator matrix
    m = X*(E*spdiags(1./sum(E,1)',0,k,k));    % compute m of each cluster
    last = label;
    [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1); % assign samples to the nearest centers
end
[~,~,label] = unique(label);
label = label(:)';
% Get cluster centers
C=[];
uLabel=unique(label);
for i=1:length(uLabel)
    currInd=find(label==uLabel(i));
    currFeat=X(:,currInd);
    C=[C mean(currFeat,2)];
end
end


