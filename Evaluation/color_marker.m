% default color palette
p_color = distinguishable_colors(40);
tmpQ = rgb2hsv(reshape(p_color,[1 size(p_color)]));
tmpQ(:,:,2) = tmpQ(:,:,2)*0.8;
tmpQ = hsv2rgb(tmpQ);
p_color = reshape(tmpQ,size(p_color));
%p_color = p_color([5:13 19:22 26 27 29 33],:); % throw r g b and black
p_color = [p_color;
       0.9098, 0.8902, 0.8902]; % add gray
% p_color = [33,87,138;
%            148,186,101;
%            39,144,176;
%            199,186,153;
%            245,135,35;
%            232,55,62;
%            139,132,183;
%            251,189,199;
%            170,31,67;
%            240,216,0;
%            14,247,248;
%            24,24,24;          
%            135,40,0;
%            232,227,227;]./255;
%default marker pool      
markerpool = ['o','>','+','h','*','x','<','s','d','^','p','v','none'];
markerpool = repmat(markerpool,1,3);