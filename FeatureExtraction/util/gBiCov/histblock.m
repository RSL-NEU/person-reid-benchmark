function lbpImg_hist =  histblock(lbpImg,ImgSize,winSize,bin_gap,win_gap)
gap_width  = win_gap(1);
gap_height = win_gap(2);
win_height = winSize(2);
win_width = winSize(1);
width  = ImgSize(1);
height = ImgSize(2);
lbpImg = reshape(lbpImg,height,width);
hist_index = 0:bin_gap:15;
numhist = length(hist_index);
lbpImg_hist =  zeros(numhist*round(width*height/gap_height/gap_width),1);
index_windows = 0;
for row = 1 : gap_height:height-win_height+1
    for col = 1 : gap_width : width-win_width+1
        X_win_temp = lbpImg(row:min((row+win_height-1),height),col:min((col+win_width-1),width));
        hist_temp = hist(X_win_temp(:),hist_index)';
        index_windows = index_windows +1;
        index_begin = (index_windows-1)* numhist+1;
        index_end =   index_windows * numhist;
        lbpImg_hist(index_begin:index_end,:) = hist_temp';
    end
end
lbpImg_hist(index_end+1:end,:) = [] ;
return;