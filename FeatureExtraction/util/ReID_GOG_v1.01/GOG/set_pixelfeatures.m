function [ num_element, lf_name, usebase ] = set_pixelfeatures( lf_type )
%  set_pixel_features
%  Set the pixel features as combinations of pre-defined base features. 
%  
%  Input: 
%           lf_type (0--yMthetaRGB, 1--yMthetaLab, 2--yMthetaHSV, 3--yMthetanRnG)
%  Output: 
%           num_element   dimension of pixel features 
%           lf_name       name of pixel features. 
%           usebase       indicator that shows which base pixel features are used

% definition of base pixel feature components ( see get_pixelfeatures.m for details)
baselfname = cell(6,1);
baselfdim = zeros(6,1);
usebase = zeros(6, 1);

baselfname{1} = 'y'; 
baselfdim(1) = 1;

baselfname{2} = 'Mtheta';
baselfdim(2) = 4;

baselfname{3} = 'RGB';
baselfdim(3) = 3;

baselfname{4} = 'LAB';
baselfdim(4) = 3;

baselfname{5} = 'HSV';
baselfdim(5) = 3;

baselfname{6} = 'rg';
baselfdim(6) = 2;

% set the pixel features as combinations of pre-defined base features. 
switch lf_type
    case 1
        lf_name = 'yMthetaRGB';
        usebase(1) = 1; % y
        usebase(2) = 1;  % Mtheta
        usebase(3) = 1; % RGB
    case 2
        lf_name = 'yMthetaLab';
        usebase(1) = 1; % y
        usebase(2) = 1;  % Mtheta
        usebase(4) = 1; % LAB
    case 3
        lf_name = 'yMthetaHSV';
        usebase(1) = 1; % y
        usebase(2) = 1;  % Mtheta
        usebase(5) = 1; % HSV
    case 4
        lf_name = 'yMthetanRnG';
        usebase(1) = 1; % y
        usebase(2) = 1;  % Mtheta
        usebase(6) = 1; % nRnG
    otherwise
        fprintf('lf_type = %d is not defined', lf_type);
end

num_element = 0;
for i=1:numel(baselfname)
    if usebase(i) == 1
        num_element = num_element + baselfdim(i);
    end
end

lf_name = strcat( lf_name );
end

