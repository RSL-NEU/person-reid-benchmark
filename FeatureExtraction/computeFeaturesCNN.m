function x=computeFeaturesCNN(I,meanImg,net)

% x=[];
% for i=1:numel(I)
%     i
%     % Subtract mean image before computing features
%     x(i,:)=computeFeaturesCNN1(I{i},net);
% end

x=[];
batchSize=16;
for i=1:batchSize:numel(I)
    i
    currBatchImgs=[];
    if(i+batchSize-1>numel(I))
        endIndx=numel(I);
    else
        endIndx=i+batchSize-1;
    end
    for j=i:endIndx
        currBatchImgs(:,:,:,j-i+1)=I{j};%-meanImg;
    end
    currBatchFeatures=computeFeaturesCNN2(currBatchImgs,net,batchSize);
    x=[x;currBatchFeatures];    
end

function x=computeFeaturesCNN1(img,net)

x1=net.forward({img});
x=reshape(x1{1,1},[1 4096]);
%x=x1{1}'; % Gives last layer
%x=x/norm(x);%
%x=net.blobs('fc8_tune').get_data; % Gives custom layer
%x=x';x=x/norm(x);
%x=x/norm(x);x=x';

function x=computeFeaturesCNN2(currBatchImgs,net,batchSize)

if(size(currBatchImgs,4)<batchSize)
    net.blobs('data').reshape(size(currBatchImgs)); 
end
x1=net.forward({currBatchImgs});
x2=net.blobs('pool5').get_data;
%x=bsxfun(@rdivide,x2,diag(sqrt(x2'*x2))'); % Make each column of x2 unit norm
%x=x2';
x=[];
for i=1:size(x2,4)
    x=[x;reshape(x2(:,:,:,i),[1 2048])];
end

