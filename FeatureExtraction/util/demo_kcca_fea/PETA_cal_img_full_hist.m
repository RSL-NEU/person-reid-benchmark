function person_rep = PETA_cal_img_full_hist(im, K_no_iso_gaus, HOG_LBP, BBoxsz, step)
    % Elyor, 07/02/2015
    % HOG_LBP = 1 means you include HOG and LBP features
    % im = image
    % K_no_iso_gaus = kernel
    % HOG_LBP: flag
    % BBoxsz = patch size(in paper [64, 16])
    % step = step size (in paper [64, 8])
    %
    % modified by Mengran Gou, 06/12/2017 
%     run('.\util\vlfeat-0.9.20\toolbox\vl_setup.m');
    im_rgb = im;
    imsz = size(im_rgb);
    imsz = imsz(1:2);
    person_rep = [];
        
%     BBoxsz = BBoxsz(end:-1:1);
%     step = step(end:-1:1);
    [~, BBox, ~] =GenerateGridBBox(imsz, BBoxsz, step);
    % Color
    for i = 1:size(BBox,1)
        im_part = im_rgb(BBox(i,2):BBox(i,4),BBox(i,1):BBox(i,3),:);
        im_hist_weight = K_no_iso_gaus(BBox(i,2):BBox(i,4),BBox(i,1):BBox(i,3),:);
        person_rep = [person_rep PETA_cal_color_histogram(im_part, im_hist_weight)];
    end
%     k=1;
%     for i = 1:8 % 1 level
%         im_part = im_rgb(k:i*16,:,:);
%         im_hist_weight = K_no_iso_gaus(k:i*16,:);
%         person_rep = [person_rep PETA_cal_color_histogram(im_part, im_hist_weight)];
%         k = i*16;
%     end
%     
    % Reduction    
    bg_h = round(imsz(1)*0.0625); % based on original stats in the paper
    bg_w = round(imsz(2)*0.125);
    im_rgb_reduced = im_rgb(bg_h+1:imsz(1)-bg_h,bg_w:imsz(2)-bg_w,:); % ###based on paper
    K_no_iso_gaus_reduced = K_no_iso_gaus(bg_h+1:imsz(1)-bg_h,bg_w:imsz(2)-bg_w,:); 
%     k=1;
%     for i = 1:7 % 2 level
%         im_part = im_rgb_reduced(k:i*16,:,:);
%         im_hist_weight = K_no_iso_gaus_reduced(k:i*16,:);
%         person_rep = [person_rep PETA_cal_color_histogram(im_part, im_hist_weight)];
%         k = i*16;
%     end
    
    if HOG_LBP == 1
        % HOG
        HOG_Fea.Params = [4 8 2 0 0.25]; % ###based on paper
        HOG_Fea   = HoG(double(im_rgb_reduced),HOG_Fea.Params)';
        person_rep = [person_rep HOG_Fea];
        
        % LBP (NOT in PAMI)
        LBP_Fea.cellSize = 16;           % ###based on paper
        LBP_Fea  = double(vl_lbp(single(rgb2gray(im_rgb_reduced)), LBP_Fea.cellSize));
%         person_rep = [person_rep reshape(LBP_Fea, [1 7*3*58])];
        person_rep = [person_rep LBP_Fea(:)'];
    end
        
    person_rep = sqrt(person_rep);
end