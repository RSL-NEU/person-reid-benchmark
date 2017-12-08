function [ F ] = LDFV(I, train, BBoxsz,step, numbin, imEdge, camID)
%[ F ] = LDFV(I, train, camID, numPatch, imEdge)
%   Reimplement code for paper 
%   B. Ma, et. al, Local Descriptors encoded by Fisher Vectors for Person
%   Re-identification, ECCVW 2012
%   I - [Nx1]cell; each cell contains all images for each person in one camera
%   train - index for the traininig person
%   numPatch - number of patches
%   numbin - number of nodes in GMM 
%   imEdge - background mask 
%   camID - [Nx1] camera ID for each image cell
%   F - LDFV feature
%
%   NOTE: Need vlfeat toolbox 
%   write by Mengran Gou

% run('.\util\vlfeat-0.9.20\toolbox\vl_setup.m');
% step =flipud([8 10; 16 16; 21 64; 64 32; 128 64]); % set the moving step size of the region.
% BBoxsz =flipud([16 21; 32 32; 22 64; 64 32; 128 64]); % set the region size.
% numP = [1 4 6 14 75];
imsz = size(I{1});
imsz = imsz(1:2);
try 
    assert(imsz(1) == 128)
catch
    error('Images must be normalized beforehand!')
end
% BBoxsz = BBoxsz(end:-1:1);
% step = step(end:-1:1);
warning off;

if ~iscell(I)
    error('Not enough input image to learn GMM!');
end
    
if ~exist('train','var')
    warning('No training index provided, all samples are used to learn GMM!');
    train = 1:numel(I);
end
if ~exist('numbin','var')
    numbin = 16;
end
if ~exist('camID','var')
    camID = ones(1,numel(I));
end
    
numChn = 17;%14+2;
% kk = find(numP == numPatch);

uCam = unique(camID); % one mask per camera
if exist('imEdge','var')
    for c = 1:numel(uCam)
        idxC = find(camID==uCam(c));
        idxC_train = intersect(train,idxC);
        % filter out background based on edge mask 
        meanEdge = cell2mat(imEdge(idxC_train));
        meanEdge = reshape(meanEdge,128,64,[]);
        meanEdge = mean(meanEdge,3);

        meanEdge(meanEdge<mean(meanEdge(:)))=0;
        meanEdge(meanEdge>0) = 1;
        se = strel('disk',5,4);
        tmpEdge = imdilate(meanEdge,se);
        tmpEdge([1 128],:) = 0;
        tmpEdge(:,[1 64]) = 0;
        tmpEdge = imfill(tmpEdge,'holes');
        meanEdge = tmpEdge;
        % meanEdge = imerode(tmpEdge,se);
        fgzone{uCam(c)} = meanEdge(:)==1;
    end
else
    uCam = unique(camID); % one mask per camera
    for c = 1:numel(uCam)
        fgzone{uCam(c)} = logical(ones(prod(imsz),1));
    end
end


% pPyramid = chnsPyramid();
% pPyramid.pChns.shrink = 1;
% pPyramid.nPerOct = 2;
% pPyramid.nApprox = 0;
% pPyramid.pChns.pColor.colorSpace = 'rgb';
% pPyramid.pChns.pGradMag.enabled = 0;


[~, BBox, region_mask] = GenerateGridBBox(imsz, BBoxsz, step);

numPatch = size(region_mask,2);

chnFeat = cell(1,numel(I));
fprintf('Begin to extract channel feature... \n');tic
for i = 1:numel(I) % per person   
    if mod(i,round(numel(I)/10))==0
            fprintf('.');
    end
    tmpI = I{i};       
    if iscell(tmpI)
        num_frame = numel(tmpI);
    else 
        num_frame = 1;
        tmpI = {tmpI};
    end
    
    if 0 %1--HOG FV;   0---LDFV
        tmpChnFeat = zeros(numChn,imsz(1)*imsz(2),num_frame, 'single');
        for f = 1:num_frame
            tmp_im = imresize(tmpI{f},imsz);
            tmp_data = {};
            % get the channel feature         
            pyramid = chnsPyramid( tmp_im, pPyramid );
            tmp_data_ch = pyramid.data{1};
            tmp_data{1} = tmp_data_ch(:,:,1:3);                 % RGB
            tmp_data{2} = rgbConvert(tmp_data{1},'luv',1);      % LUV
            tmp_data{3} = rgbConvert(tmp_data{1},'hsv',1);      % HSV
            tmp_data{4} = tmp_data_ch(:,:,4:9);                 % Gradient
            tmp_data = cat(3,tmp_data{:});
            tmp_data(:,:,6) = []; % remove duplicated V channel
            tmp_data(:,:,15) = repmat([1:imsz(1)]',1,imsz(2));   % x,y location
            tmp_data(:,:,15) = tmp_data(:,:,15)./imsz(1);
            tmp_data(:,:,16) = repmat([1:imsz(2)],imsz(1),1);
            tmp_data(:,:,16) = tmp_data(:,:,16)./imsz(2);
            
            tmp_data = permute(tmp_data, [3 1 2]);            
            tmp_data = reshape(tmp_data,size(tmp_data,1), []);
            tmpChnFeat(:,:,f) = tmp_data;
        end
    else 
        tmpChnFeat = zeros(numChn,imsz(1)*imsz(2),num_frame, 'single');
        for f = 1:num_frame
            tmp_im = imresize(tmpI{f},imsz);
            tmp_data = zeros(imsz(1),imsz(2),17);
            % get the channel feature
            tmp_data(:,:,1) = repmat([1:imsz(1)]',1,imsz(2));
            tmp_data(:,:,1) = tmp_data(:,:,1)./imsz(1);
            tmp_data(:,:,2) = repmat([1:imsz(2)],imsz(1),1);
            tmp_data(:,:,2) = tmp_data(:,:,2)./imsz(2);
            tmp_hsv = rgb2hsv(tmp_im);
            tmp_data(:,:,3:5) = tmp_hsv;
            for c = 1:3
                [tmpX,tmpY] = imgradientxy(reshape(tmp_hsv(:,:,c),imsz(1),imsz(2)));
                [tmpXX,~] = imgradientxy(tmpX);
                [~,tmpYY] = imgradientxy(tmpY);
                tmp_data(:,:,(c-1)*4+6:c*4+5) = cat(3,tmpX,tmpY,tmpXX,tmpYY);
            end
%             tmp_data = tmp_data(:,:,[1 2 6:17]); % only use edge part
%             tmp_data = tmp_data(:,:,1:5); % only use color
            tmp_data = permute(tmp_data, [3 1 2]);
            tmp_data = reshape(tmp_data,size(tmp_data,1),[]);
            tmpChnFeat(:,:,f) = tmp_data;
        end
    end
    chnFeat{i} = tmpChnFeat;
end
numChn = size(chnFeat{1},1);
fprintf('Done!\n');toc
fprintf('Begin to build GMM model...\n');tic
% GMM encoding
idx = 1:numel(I);
idx_train = ismember(idx,train);
for s = 1:numPatch   
    ChnFeat_patch = [];
    for c = 1:numel(unique(camID))
        tmpChnFeat = cellfun(@(x) x(:,logical(region_mask(:,s) & fgzone{uCam(c)}),:), chnFeat(idx_train & camID==uCam(c)),'UniformOutput',0);
        tmpChnFeat = cellfun(@(x) reshape(x,numChn,[]), tmpChnFeat,'UniformOutput',0);
        tmpChnFeat = cell2mat(tmpChnFeat);
        % subsample
        MAXsample = 100000;
        idxsub = randsample(size(tmpChnFeat,2),min(MAXsample,size(tmpChnFeat,2))); 
        tmpChnFeat = tmpChnFeat(:,idxsub);
        ChnFeat_patch = [ChnFeat_patch tmpChnFeat];
    end
    [means{s}, covariances{s}, priors{s}] = vl_gmm(ChnFeat_patch,numbin,'NumRepetitions',1);    
end
fprintf('Done!\n');toc
fprintf('Begin to extract fisher vectors...\n');tic
% FV encode
F = zeros(numel(chnFeat),numChn*2*numbin*numPatch,'single');
for i = 1:numel(chnFeat)
    if mod(i,round(numel(I)/10))==0
            fprintf('.');
    end
    tmpChnFeat = chnFeat{i};
    tmpF_perP = zeros(numChn*2*numbin*numPatch,size(tmpChnFeat,3),'single');
    for f = 1:size(tmpChnFeat,3)
        tmpFrame = tmpChnFeat(:,:,f);
        tmpF_perf = zeros(numChn*2*numbin,numPatch,'single');
        for s = 1:numPatch        
            tmpFrame_s = tmpFrame(:,logical(region_mask(:,s)) & fgzone{camID(i)});
            tmpF_perf(:,s) = vl_fisher(tmpFrame_s,means{s}, covariances{s}, priors{s},'Normalized','SquareRoot');
        end
        tmpF_perP(:,f) = tmpF_perf(:);
    end
    % naive mean along the temporal dimension
    tmpF_mean = mean(tmpF_perP,2);
%     % normalize on each FV
%     tmpF_mean = sign(tmpF_mean).*(sqrt(abs(tmpF_mean)));    % power
%     FVdim = numChn*2*numbin;
%     for s = 1:numPatch        
%         tmpF_mean((s-1)*FVdim+1:s*FVdim) = normc_safe(tmpF_mean((s-1)*FVdim+1:s*FVdim)); % L2
%     end
    tmpF_mean(isnan(tmpF_mean)) = 0;
    F(i,:) = tmpF_mean';
end
fprintf('Done!\n');toc
% F = sign(F).*(sqrt(abs(F)));
% F = normc_safe(F');
% F = F';




