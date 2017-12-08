function CR = findCovarianceMatrix(I, positionX, positionY, winSize)
% compute the covariance of the region in the image.
% Input:
%       I: the matrix of the image
%       positionX: the x coordinate of the left-upper corner of the region
%       positionY: the y coordinate of the left-upper corner of the region
%       winSize: the size of region.
% Output:
%       CR: the covariance matrix of the region

% * Current Version£º1.1
% * Replace Version£º1.0
% * Author£ºBingpeng MA
% * Date£º2012-01-25
% * Revision: the region can be the  rectangle, not the square.

% First and second derivatives
d = [-1 0 1];
dd = [-1 2 -1];
dI = double(I);

Ix = conv2(d, dI);
Iy = conv2(d,1,dI);
Ixx = conv2(dd, dI);
Iyy = conv2(dd,1,dI);

[height, width] = size(I);
win_width = winSize(1);
win_height = winSize(2);
X = positionX -1 + (1: win_width);
X = repmat(X, 1, win_height);
X = X(:);
Y = positionY -1 + (1: win_height);
Y = repmat(Y, win_width,1);
Y = Y(:);
index_temp = (X-1)*height + Y;
index_temp1 = (X-1)*(height+2) + Y;
f = [X  Y  dI(index_temp)  Ix(index_temp)   Iy(index_temp1)  Ixx(index_temp)  Iyy(index_temp1)];

uR = mean(f);
[num,dim] = size(f);
f = f - repmat(uR,num,1);
T = zeros(dim, dim);
win_widhei = win_width * win_height;
for k = 1:win_widhei
    temp = f(k,:)'*f(k,:);
    T = T + temp;
end
CR = (1/ win_widhei)*T;
return;