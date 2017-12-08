clear all;
clc;
load viper_demo_data;
m_nScale = 24;
% ImgData1, ImgData2, ImgData3, are the data of H, S and V channels, respectively.
ImgData1 = gBiCov(ImgData1,ImgSize,winSize,win_gap,m_nScale,8);
ImgData2 = gBiCov(ImgData2,ImgSize,winSize,win_gap,m_nScale,8);
ImgData3 = gBiCov(ImgData3,ImgSize,winSize,win_gap,m_nScale,8);


dim = size(ImgData1,1)/2;
ImgData_bicov = [ImgData1(1:dim,:); ImgData2(1:dim,:); ImgData3(1:dim,:)] ;
ImgData_gabor = [ImgData1(1+dim:end,:); ImgData2(1+dim:end,:); ImgData3(1+dim:end,:)];
for cross = 1 : 2
    temp = ImgData_bicov(:, cross);
    temp = sign(temp).*(abs(temp)).^0.5; % power
    temp = temp/norm(temp); % L2 
    ImgData_bicov(:, cross) = temp;
    
    temp = ImgData_gabor(:, cross);
    temp = sign(temp).*(abs(temp)).^0.5; % power
    temp = temp/norm(temp); % L2
    ImgData_gabor(:, cross) = temp;
end
ImgData_gBiCov = [ImgData_bicov; ImgData_gabor];

 