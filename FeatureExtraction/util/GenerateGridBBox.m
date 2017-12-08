% Generate Grid Bounding Box
% calculate RGB, LAB, LOG-RGB and Gabor feature mapping of the input image.
% By Fei Xiong, 
%    ECE Dept, 
%    Northeastern University 
%    2013-11-04
% INPUT
% BBoxsz: Bounding Box size [height, width]
% imsz: image size [height, width]
% OUTPUT
% region_idx: the cell structure storing the index of pixel for each
% BBox: the BoundingBox positions. Each row is in the format of 
%       [xtop, ytop, xbottom, ybottom];
function [region_idx, BBox, region_mask] =GenerateGridBBox(imsz, BBoxsz, step)
BBox=[];
BBoxsz = min(BBoxsz,imsz);
step = min(BBoxsz,step);
ytop = floor(1:step(1):imsz(1)-BBoxsz(1)+1)';
xtop = floor(1:step(2):imsz(2)-BBoxsz(2)+1);
gridy = length(ytop);
gridx = length(xtop);
ytop = repmat(ytop, 1, gridx);
xtop = repmat(xtop, gridy, 1);
BBox = [xtop(:), ytop(:), xtop(:)+BBoxsz(2)-1, ytop(:)+BBoxsz(1)-1];
region_mask = zeros(imsz(1)*imsz(2), size(BBox,1));
for i =1: size(BBox,1)
    temp = zeros(imsz);
    temp(BBox(i,2):BBox(i,4), BBox(i,1):BBox(i,3)) =1;
    region_idx{i,1} = find(temp(:)); % pixel index for each region
    region_mask(:,i) = temp(:);
end
return;