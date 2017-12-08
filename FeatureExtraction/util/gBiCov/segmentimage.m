function [newImgData, pos] = segmentimage(ImgData,Imgsize,Pos)
 

x = Pos(1);
y = Pos(2);
subwidth = Pos(3);
subheight = Pos(4);
height = Imgsize(2);
width = Imgsize(1);
col_end = min(width, x + subwidth - 1) ;
subheight = min(subheight, height - y +1);
pos = [];
for cross_col =  x : col_end
    dim_begin =  (cross_col- 1)* height + y;
    dim_end = dim_begin  + subheight-1;
    pos= [pos dim_begin : dim_end] ;
end
newImgData = ImgData(pos,:);

return;