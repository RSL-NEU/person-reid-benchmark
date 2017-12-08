function pos_all = imageblockindex(sub_width,sub_height, gap,width, height)

sub_heiwid = sub_height * sub_width;
index = 0 ;ImgSize = [height width] ;
ImgData = zeros(width*height,1);
pos_all = zeros(sub_width*sub_height*height/gap*width/gap,1);
for cross_x = 1:gap:width-sub_width+1
    for cross_y = 1: gap : height-sub_height+1
        index= index +1;
        Pos = [cross_x cross_y sub_height sub_width] ;
        [newImgData, pos] = segmentimage(ImgData,ImgSize,Pos);
        index_begin = (index - 1) * sub_heiwid +1;
        index_end = index * sub_heiwid;
        pos_all(index_begin:index_end)  = pos ;
    end
end
dim_pos = index_end;
pos_all(dim_pos+1:end)= [] ;
clear index_begin  index_end pos newImgData index;
return;