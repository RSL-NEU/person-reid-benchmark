%
% compute dense color feature for an image
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

function [color_arr, grid_x, grid_y] = sp_dense_color(I, options)
% dense color MATLAB script
%

grid_spacing = options.gridspacing;
patch_size = options.patchsize;
num_bins = options.nbins;
scale = options.scale;
sigma = options.sigma;
clamp = options.clamp;

num_scales = numel(scale);
I = rgb2lab_dense(I);

epsi = 1e-8;
[hgt, wid, dimcolor] = size(I);
Nx = length(patch_size(1)/2:grid_spacing(1):wid-patch_size(1)/2);
Ny = length(patch_size(2)/2:grid_spacing(2):hgt-patch_size(2)/2);

color_arr = zeros(num_bins, Nx*Ny, num_scales, dimcolor);

for v = 1:dimcolor
    for s = 1:num_scales
        if scale(s) ~= 1
            G = gen_gauss(sigma/scale(s));
            I_filtered = conv2(I(:, :, v), G, 'same');
        else 
            I_filtered = I(:, :, v);
        end
        x = round(linspace(1, wid, round(wid*scale(s))));
        y = round(linspace(1, hgt, round(hgt*scale(s))));
        [X, Y] = meshgrid(x, y);
        I_scaled = interp2(I_filtered, X, Y, 'nearest');

        patch_size_s = patch_size*scale(s);
        halfpatch = (patch_size_s-1)/2;
        [hgt_s, wid_s] = size(I_scaled);
        
        % grid        
        grid_x = ceil(linspace(patch_size_s(1)/2, wid_s-patch_size_s(1)/2, Nx));
        grid_y = ceil(linspace(patch_size_s(2)/2, hgt_s-patch_size_s(2)/2, Ny));
        
        X = repmat(grid_x, Ny, 1);
        Y = repmat(grid_y', 1, Nx);
        grid = [X(:), Y(:)];
        
        % extract dense color histogram
        scale_s = scale(s);
        parfor i = 1:Nx*Ny
            hist = colorHist(imcrop(I_scaled, ...
                [grid(i, :)-halfpatch+1, patch_size_s-1]), num_bins, v);
            hist = hist./scale_s;
            % L2-clamp norm
            norm_tmp = hist/sqrt(sum(hist.^2)+epsi^2);
            norm_tmp(norm_tmp >= clamp) = clamp;
            norm_tmp = norm_tmp/sqrt(sum(norm_tmp.^2)+epsi^2);
            color_arr(:, i, s, v) = norm_tmp;
        end
    end
end
grid_x = ceil(linspace(patch_size(1)/2, wid-patch_size(1)/2, Nx));
grid_y = ceil(linspace(patch_size(2)/2, hgt-patch_size(2)/2, Ny));
[grid_x,grid_y] = meshgrid(grid_x, grid_y);


function cvt = rgb2lab_dense(img)
cvt = zeros(size(img));
img = double(img);
cvt(:, :, 1) = (img(:, :, 1) - img(:, :, 2))./sqrt(2); % + 255/sqrt(2);
cvt(:, :, 2) = (img(:, :, 1) + img(:, :, 2) - 2.*img(:, :, 3))./sqrt(6); % + 510/sqrt(6);
cvt(:, :, 3) = (img(:, :, 1) + img(:, :, 2) + img(:, :, 3))./sqrt(3);


function h = colorHist(cvt, K, dim)
%compute the color histogram of converted image
if dim == 1
    mini = -255/sqrt(2);
    maxi = 255/sqrt(2);
elseif dim == 2
    mini = -510/sqrt(6);
    maxi = 510/sqrt(6);
else
    mini = 0;
    maxi = 765;
end
cvt = floor((K+1e-8)*(cvt - mini)./(maxi-mini)) + 1;
label = min(K, cvt(:));
h = hist(label, 1:K);
h = h';


function G=gen_gauss(sigma)

if all(size(sigma)==[1, 1])
    % isotropic gaussian
	f_wid = 4 * ceil(sigma) + 1;
    G = fspecial('gaussian', f_wid, sigma);
%	G = normpdf(-f_wid:f_wid,0,sigma);
%	G = G' * G;
else
    % anisotropic gaussian
    f_wid_x = 2 * ceil(sigma(1)) + 1;
    f_wid_y = 2 * ceil(sigma(2)) + 1;
    G_x = normpdf(-f_wid_x:f_wid_x,0,sigma(1));
    G_y = normpdf(-f_wid_y:f_wid_y,0,sigma(2));
    G = G_y' * G_x;
end


