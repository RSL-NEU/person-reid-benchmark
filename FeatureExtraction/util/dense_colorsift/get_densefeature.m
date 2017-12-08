%
% extract dense color histogram and dense SIFT features and concatenate
% them
%
% Created by Rui Zhao, on May 20, 2013. 
% This code is release under BSD license, 
% any problem please contact Rui Zhao rzhao@ee.cuhk.edu.hk
%
% Please cite as
% Rui Zhao, Wanli Ouyang, and Xiaogang Wang. Unsupervised Salience Learning
% for Person Re-identification. In IEEE Conference of Computer Vision and
% Pattern Recognition (CVPR), 2013. 
%

function dfeat = get_densefeature(I, options_color, options_sift, nynx)
% dense color [nbins=32][nx*ny][nscales=3][ncolor=3]
dcolor_tmp = sp_dense_color(I, options_color);
dsift_tmp = reshape(sp_dense_sift(I, options_sift), 128, nynx, options_sift.color);

dfeat = zeros(options_color.nbins*3*3 + 128*3, nynx);
for i = 1:nynx
    dfeat(:, i) = [reshape(dcolor_tmp(:, i, :, :), [], 1); ...
        reshape(dsift_tmp(:, i, :), [], 1)];
end

% apply PCA to reduce dimensionalitiy??
