function X_gBiCov = gBiCov(ImgData,ImgSize,winSize,win_gap,m_nScale,m_nOrientation)
% Input£º
%       Img: the data of the input image, a matrix£¬the size is [height*width£¬num]£¬
%       ImgSize£º the size of image£¬width = ImgSize(1);height = ImgSize(2);
%       winSize: the size of windows [x,y].
%       win_gap: the gap of windows [x,y].
%       m_nScale: the number of Gabor scales
%       m_nOrientation: the number of Gabor orientation.
% Output:
%       BifFeature£ºthe feature of gBiCov

% * Current Version£º1.0
% * Author£ºBingpeng MA
% * Date£º2011-08-25

if nargin<3
    winSize = [8 8] ;
end
if nargin<4
    win_gap = fix(winSize/2);
end
if nargin<5
    m_nScale = 16;
end
if nargin<6
    m_nOrientation = 8;
end

[dim, num] = size(ImgData);
m_nOrientation = 8;
width = ImgSize(1);
height = ImgSize(2);
for cross_num = 1 : num
    disp(['The features of gBiCov are extracting ' num2str(cross_num)]);
    ImgData_temp = ImgData(:,cross_num );
    GaborFeature = gaborextractionmultiscale(ImgData_temp,ImgSize,m_nScale,m_nOrientation);
    clear ImgData_temp;
    GaborFeature_temp = zeros(height*width, m_nScale/2);
    for cross = 1 : m_nScale/2
        index = ((cross-1)*m_nOrientation + 1) : cross*m_nOrientation;
        GaborFeature_temp(:, cross) = mean(GaborFeature(:, index), 2);
    end
    GaborFeature = GaborFeature_temp;
    clear GaborFeature_temp cross index;
    X1 = GaborFeature(:, 1: 2:end);
    X2 = GaborFeature(:, 2: 2:end);
    clear GaborFeature ;
    for cross_scale = 1 : m_nScale/4
%         win_gap = winSize/2;
        X_cov_temp = computedistance(X1(:, cross_scale),X2(:, cross_scale),ImgSize,winSize,win_gap);
        X_gabor_temp = computemeanoffeature(X1(:, cross_scale),X2(:, cross_scale),ImgSize,winSize,win_gap)*100;
        if cross_scale == 1
            dim_scale = length(X_cov_temp);
            X_cov = zeros(dim_scale, m_nScale/4);
            X_gabor = X_cov ;
        end
        X_cov(:,cross_scale) =  X_cov_temp;
        X_gabor(:,cross_scale) =  X_gabor_temp;
        clear  X_cov_temp X_gabor_temp;
    end
    clear X1 X2 cross_scale;
    X_cov = X_cov(:);
    X_gabor = X_gabor(:);
    if cross_num == 1
        dim_scale = length(X_cov);
        X_gBiCov = zeros(dim_scale*2,num);
    end
    X_gBiCov(:, cross_num) = [X_cov; X_gabor];
    clear X_cov X_gabor;
end
return;