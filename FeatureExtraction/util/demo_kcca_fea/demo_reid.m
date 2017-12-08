%%
% To run this code you have to mex
% + HoG.cpp
% + Include vl_feat library: http://www.vlfeat.org/download.html
%   Every images takes less than 0.2 s
  

%%%%%%%% 1. FEATURE EXTRACTION %%%%%%%%
load('VIPeR_Images.mat');
load('K_no_iso_gaus.mat');
VIPeR_kcca_features = zeros(1264,5138);
for im_id = 1:1264        % the number of samples
    im_id
    im = I{im_id};
    VIPeR_kcca_features(im_id,:) = PETA_cal_img_full_hist(im,K_no_iso_gaus,1);
end

%%
% %%%%%%%%%% 2. Testing %%%%%%%%%%
test_a = VIPeR_kcca_features(634:2:end, :)';
test_b = VIPeR_kcca_features(633:2:end, :)';

num_p          = 316;
num_gallery    = 316;
num_test       = num_p;
idxProbe       = 1:316;  test_a_after_D = test_a(:,idxProbe);    
idxGallery     = 1:316;  test_b_after_D = test_b(:,idxGallery);   

scores_after    = pdist2(test_b_after_D', test_a_after_D','cosine');
cmc_nn          = zeros(num_gallery,3);
cmc_nn(:,1)     = 1:num_gallery;
cmcCurrent      = zeros(num_gallery,3);
cmcCurrent(:,1) = 1:num_gallery;

for k=1:num_test
    finalScore = scores_after(:,k);
    [sortScore sortIndex] = sort(finalScore);
    [cmc_nn, ~] = evaluateCMC_demo(idxProbe(k),idxGallery(sortIndex),cmc_nn,cmcCurrent);
end
ranks  = cmc_nn(: ,2)./cmc_nn(: ,3);
rank_1 = ranks(1)
figure(1);hold on;plotCMCcurve(cmc_nn,'b','','VIPeR');



