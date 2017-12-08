function X_dist = computedistance(X1, X2,ImgSize,winSize,win_gap)
% Compute the covariance distance of two images
% Input:
%       X1: the first image£¬a matrix of [dim, 1]
%       X2: the second image£¬a matrix of [dim, 1]
%       ImgSize:   the size of image£¬width = ImgSize(1);height = ImgSize(2);
%       winSize: the size of windows.
%       win_gap: the gap of windows.
% Output:
%       X_dist: the distance of X1 and X2, is a vector.

% * Current Version£º1.0
% * Author£ºBingpeng MA
% * Date£º2011-08-25

gap_width  = win_gap(1);
gap_height = win_gap(2);
win_height = winSize(2);
win_width = winSize(1);
width  = ImgSize(1);
height = ImgSize(2);
X1 = reshape(X1, height, width);
X2 = reshape(X2, height, width);
X_dist =  zeros(round(1.2*width*height/gap_height/gap_width),1);
index = 0;
for row = 1 : gap_height:height-win_height+1
    for col = 1 : gap_width : width-win_width+1
        CR = findCovarianceMatrix(X1, col, row, winSize) ;
        CcR = findCovarianceMatrix(X2, col, row, winSize) ;
        % distance metric
        temp = eig(CR,CcR);
        index_temp = find(temp>0 & temp < 1e10);
        temp = temp(index_temp);
        temp = sqrt(sum(log(temp).^2));
        index = index +1;
        X_dist(index,:) = temp;
    end
end
X_dist(index+1:end,:) = [];
return;