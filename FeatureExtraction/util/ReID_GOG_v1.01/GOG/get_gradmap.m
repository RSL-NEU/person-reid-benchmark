function [ qori, ori, mag ] = get_gradmap( X , binnum )
% X -- input image. Size [h w 1]
% binnum -- number of graient orientation bins
% qori -- quantized gradient orientation map (soft voting). Size [h w binnum]
% ori -- graident orientation map. Size [h w 1]
% mag -- gradient magnitude map. Size [h w 1]

% gradient filter
hx = [-1,0,1];
hy = -hx';
grad_x = imfilter( double(X), hx);
grad_y = imfilter( double(X), hy);

ori = (atan2( grad_x, grad_y) + pi)*180/pi; % gradient orientations
mag = sqrt(grad_x.^2 + grad_y.^2 ); % gradient magnitude

binwidth = 360/binnum;
IND = floor( ori./binwidth);

ref1 = IND.*binwidth;
ref2 = (IND + 1).*binwidth;

dif1 = ori - ref1;
dif2 = ref2 - ori;

weight1 = dif2./(dif1 + dif2);
weight2 = dif1./(dif1 + dif2);

[h w] = size(X);
qori = zeros(h, w, binnum);

IND(IND == binnum) = 0;

IND1 = IND + 1;
IND2 = IND + 2;
IND2(IND2 == binnum + 1) = 1;

for y=1:h
    for x=1:w
        qori(y,x,IND1(y,x)) = weight1(y,x);
        qori(y,x,IND2(y,x)) = weight2(y,x);
    end
end

end

