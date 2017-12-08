function [ F ] = get_pixelfeatures( X, lfparam )
%   get_pixelfeaturres.m 
%   extract pixelfeatures map defined by lfparam
%
%   Input: 
%          <X>: input RGB image. Size: [h, w, 3]
%          [lfparam]: parameters.
%
%   Output: 
%          <F>: pixel feature map. Size: [h, w, lfparam.num_element]
%
%
[h, w, col] = size(X);
F = zeros(h,w,lfparam.num_element);
dimstart = 1;

if lfparam.usebase(1) == 1 % y
    PY = zeros( size(X,1), size(X,2));
    for tmpy = 1:size(X,1)
        PY(tmpy,:) = tmpy;
    end
    PY = double(PY)./size(X,1);
    
    F(:,:,dimstart) = PY;
    dimstart = dimstart + 1;
end
if lfparam.usebase(2) == 1 % M_theta
    binhog = 4;
    img_ycbcr = rgb2ycbcr(uint8(X));
    img_ycbcr = double(img_ycbcr);
    img_ycbcr(:,:,1) = (img_ycbcr(:,:,1) - 16)./235;
    
    [qori, ori, mag] = get_gradmap( img_ycbcr(:,:,1), binhog);
    Yq = qori.*repmat( mag, [1, 1, binhog]);
    
    F(:,:, dimstart: dimstart + binhog-1) = Yq;
    dimstart = dimstart + binhog;
end
if lfparam.usebase(3) == 1 % RGB
    F(:,:,dimstart:dimstart+2) = double(X)./255;
    dimstart = dimstart + 3;
end
if lfparam.usebase(4) == 1 % LAB
    img_lab = rgb2lab(X);
    img_lab(:,:,1) = img_lab(:,:,1)./100;
    img_lab(:,:,2) = (img_lab(:,:,2) + 50)./100;
    img_lab(:,:,3) = (img_lab(:,:,3) + 50)./100;
    F(:,:, dimstart: dimstart+2) = img_lab;
    dimstart = dimstart + 3;
end
if lfparam.usebase(5) == 1 % HSV
    F(:,:,dimstart:dimstart+2) = rgb2hsv(X);
    dimstart = dimstart + 3;
end
if lfparam.usebase(6) == 1 % nRnG
    sumVal = max(sum(X, 3), 1);
    F(:,:,dimstart) = double(X(:, :, 1))./sumVal;
    F(:,:,dimstart + 1) = double(X(:, :, 2))./sumVal;
    dimstart = dimstart + 2;
end

end

