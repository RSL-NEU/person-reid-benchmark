%
% compute dense SIFT feature for an image
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

function [sift_arr, grid_x, grid_y] = sp_dense_sift(I, options)%grid_spacing, patch_size , sigma_edge)

% I                          = imread('image_0174.jpg');
% [sift_arr, grid_x, grid_y] = sp_dense_sift(I, 4 , 9);
%
%
% Original script by Svetlana Lazebnick
% Adapted by Antonio Torralba: modified using convolutions to speed up the computations.
% And brought back into Svetlana's library

% if(~exist('grid_spacing','var'))
%     grid_spacing = 1;
% end
% if(~exist('patch_size','var'))
%     patch_size = 16;
% end

%I = rgb2gray(I);

% I = double(I);
% I = mean(I,3);
% I = I /max(I(:));

grid_spacing = options.gridspacing;
patch_size = options.patchsize;

if options.color == 1 % gray scale
    im = rgb2gray(I);
elseif options.color == 2 % rgb
    im = I;
else
    im = rgb2lab(I);
end

[hgt, wid, ~] = size(I);
Nx = length(patch_size(1)/2:grid_spacing(1):wid-patch_size(1)/2);
Ny = length(patch_size(2)/2:grid_spacing(2):hgt-patch_size(2)/2);
grid_x = ceil(linspace(patch_size(1)/2, wid-patch_size(1)/2, Nx));
grid_y = ceil(linspace(patch_size(2)/2, hgt-patch_size(2)/2, Ny));
 
% parameters
num_angles = 8;
num_bins = 4;
num_samples = num_bins * num_bins;
alpha = 9; %% parameter for attenuation of angles (must be odd)

if nargin < 4
    sigma_edge = 1;
end

angle_step = 2 * pi / num_angles;
angles = 0:angle_step:2*pi;
angles(num_angles+1) = []; % bin centers

[G_X,G_Y]=gen_dgauss(sigma_edge);

% sift_arr = zeros([length(grid_y) length(grid_x) num_angles*num_bins*num_bins], 'single');
sift_arr = zeros([num_angles*num_bins*num_bins, length(grid_y), length(grid_x), options.color], 'single');

for v = 1:options.color
    I = double(im(:, :, v));
    
    single_arr = zeros([length(grid_y) length(grid_x) num_angles*num_bins*num_bins], 'single');
    % add boundary:
    % I = [I(2:-1:1,:,:); I; I(end:-1:end-1,:,:)];
    % I = [I(:,2:-1:1,:) I I(:,end:-1:end-1,:)];
    
    %I = I-mean(I(:));
    I_X = filter2(G_X, I, 'same'); % vertical edges
    I_Y = filter2(G_Y, I, 'same'); % horizontal edges
    
    % I_X = I_X(3:end-2,3:end-2,:);
    % I_Y = I_Y(3:end-2,3:end-2,:);
    
    I_mag = sqrt(I_X.^2 + I_Y.^2); % gradient magnitude
    I_theta = atan2(I_Y,I_X);
    
    %I_theta(find(isnan(I_theta))) = 0; % necessary????
    
    % grid
    % grid_x = patch_size/2:grid_spacing:wid-patch_size/2+1;
    % grid_y = patch_size/2:grid_spacing:hgt-patch_size/2+1;
    
    % make orientation images
    I_orientation = zeros([hgt, wid, num_angles], 'single');
    
    % for each histogram angle
    cosI = cos(I_theta);
    sinI = sin(I_theta);
    for a=1:num_angles
        % compute each orientation channel
        tmp = (cosI*cos(angles(a))+sinI*sin(angles(a))).^alpha;
        tmp = tmp .* (tmp > 0);
        
        % weight by magnitude
        I_orientation(:,:,a) = tmp .* I_mag;
    end
    
    % Convolution formulation:
    %weight_kernel = zeros(patch_size,patch_size);
    r = patch_size/2;
    cx = r - 0.5;
    sample_res = patch_size/num_bins;
    weight_x = abs((1:patch_size(1)) - cx(1))/sample_res(1);
    weight_x = (1 - weight_x) .* (weight_x <= 1);
    
    weight_y = abs((1:patch_size(2)) - cx(2))/sample_res(2);
    weight_y = (1 - weight_y) .* (weight_y <= 1);
    
    for a = 1:num_angles
        %I_orientation(:,:,a) = conv2(I_orientation(:,:,a), weight_kernel, 'same');
        I_orientation(:,:,a) = conv2(weight_x, weight_y', I_orientation(:,:,a), 'same');
    end
    
    % Sample SIFT bins at valid locations (without boundary artifacts)
    % find coordinates of sample points (bin centers)
    %[sample_x, sample_y] = meshgrid(linspace(1,patch_size+1,num_bins+1));
    [sample_x, sample_y] = meshgrid(linspace(1,patch_size(1),num_bins+1),...
                                    linspace(1,patch_size(2),num_bins+1));
    
    sample_x = sample_x(1:num_bins,1:num_bins);
    % sample_x = sample_x(:)-patch_size/2;
    sample_x = ceil(sample_x(:)-patch_size(1)/2);
    
    sample_y = sample_y(1:num_bins,1:num_bins);
    % sample_y = sample_y(:)-patch_size/2;
    sample_y = ceil(sample_y(:)-patch_size(2)/2);
    
    b = 0;
    for n = 1:num_bins*num_bins
        single_arr(:,:,b+1:b+num_angles) = I_orientation(grid_y+sample_y(n), grid_x+sample_x(n), :);
        b                              = b+num_angles;
    end
    clear I_orientation

    % normalize SIFT descriptors
    %[nrows, ncols, cols] = size(sift_arr);
    %sift_arr = reshape(sift_arr, [nrows*ncols num_angles*num_bins*num_bins]);
    %sift_arr = normalize_sift(sift_arr);
    %sift_arr = reshape(sift_arr, [nrows ncols num_angles*num_bins*num_bins]);
     
    ct       = .000001;
    single_arr = single_arr + ct;
    tmp      = sqrt(sum(single_arr.^2, 3));
    single_arr = single_arr ./ repmat(tmp, [1 1 size(single_arr,3)]);
    
    sift_arr(:, :, :, v) = shiftdim(single_arr, 2);
    
end

% Outputs:
[grid_x,grid_y]      = meshgrid(grid_x, grid_y);


function [GX,GY]=gen_dgauss(sigma)

% laplacian of size sigma
%f_wid = 4 * floor(sigma);
%G = normpdf(-f_wid:f_wid,0,sigma);
%G = G' * G;
G = gen_gauss(sigma);
[GX,GY] = gradient(G); 

GX = GX * 2 ./ sum(sum(abs(GX)));
GY = GY * 2 ./ sum(sum(abs(GY)));


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


function Lab = rgb2lab(img)
% convert RGB image into Lab space

cform = makecform('srgb2lab');
Lab = applycform(img, cform);
Lab = double(Lab);

%do normalization on each channel, such that the range of each
%channel is [0 1]. In this dataset, the maximum value of the L
%channel is 255, and the minimum is 0; the maximum value of the a
%channel is 187 and the minimum is 108; the maximum value of the b
%channel is 161 and the minimum is 81.

Lab(:,:,1) = Lab(:,:,1)/255;
Lab(:,:,2) = (Lab(:,:,2)-108)/(187-108);
Lab(:,:,3) = (Lab(:,:,3)-81)/(161-81);

