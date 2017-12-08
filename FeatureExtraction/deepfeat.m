function x = deepfeat(img,net,batchSz)
% extract CNN feature based on pre-trained image net model
cf = pwd;
cd('../Misc/')
run('matconvnet/matlab/vl_setupnn.m');
normSz = net.meta.normalization.imageSize(1:2);

idx_start = 1;
while idx_start <= numel(img)
    % load batch images
    idx_end = min(idx_start+batchSz-1, numel(img));
    img_counter = 1;
    im_ = zeros(normSz(1),normSz(2),3,idx_end-idx_start+1,'single');
    for i = idx_start:idx_end
        im_(:,:,:,img_counter) = imresize(img{i},normSz);
        img_counter = img_counter + 1;
    end
    % remove mean
    im_ = bsxfun(@minus,im_,net.meta.normalization.averageImage) ;
    % test on CNN
    res = vl_simplenn(net, im_,[],[],'mode','test');
    % get fc7 feature
    x(:,idx_start:idx_end) = squeeze(res(end).x);
    idx_start = idx_end + 1;
end
x = normc_safe(x);
cd(cf)