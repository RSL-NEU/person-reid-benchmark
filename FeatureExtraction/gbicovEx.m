function feature_gB = gbicovEx(img,patchsize,gridstep)

patchsize = patchsize(end:-1:1);
gridstep = gridstep(end:-1:1);

m_nScale = 24;
[h,w,~] = size(img{1});
im_ch1 = zeros(h*w, numel(img));
im_ch2 = zeros(h*w, numel(img));
im_ch3 = zeros(h*w, numel(img));
for i = 1 : numel(img)
    img{i} = rgb2hsv(img{i});
    im_ch1(:,i) = reshape(img{i}(:,:,1),[],1);
    im_ch2(:,i) = reshape(img{i}(:,:,2),[],1);
    im_ch3(:,i) = reshape(img{i}(:,:,3),[],1);
end
im_ch{1} = im_ch1;
im_ch{2} = im_ch2;
im_ch{3} = im_ch3;

for c = 1:3
    gB_ch{c} = gBiCov(im_ch{c},[w,h],patchsize, gridstep, m_nScale, 8);
end
gB_ch1 = gB_ch{1};
gB_ch2 = gB_ch{2};
gB_ch3 = gB_ch{3};
% gB_ch1 = gBiCov(im_ch1,[w,h],patchsize, gridstep, m_nScale, 8);
% gB_ch2 = gBiCov(im_ch2,[w,h],patchsize, gridstep, m_nScale, 8);
% gB_ch3 = gBiCov(im_ch3,[w,h],patchsize, gridstep, m_nScale, 8);

dim = size(gB_ch1,1)/2;
ImgData_bicov = [gB_ch1(1:dim,:); gB_ch2(1:dim,:); gB_ch3(1:dim,:)] ;
ImgData_gabor = [gB_ch1(1+dim:end,:); gB_ch2(1+dim:end,:); gB_ch3(1+dim:end,:)];
for cross = 1 : size(ImgData_bicov,2)
    temp = ImgData_bicov(:, cross);
    temp = sign(temp).*(abs(temp)).^0.5; % power
    temp = temp/norm(temp); % L2 
    ImgData_bicov(:, cross) = temp;
    
    temp = ImgData_gabor(:, cross);
    temp = sign(temp).*(abs(temp)).^0.5; % power
    temp = temp/norm(temp); % L2
    ImgData_gabor(:, cross) = temp;
end
feature_gB = [ImgData_bicov; ImgData_gabor];

 