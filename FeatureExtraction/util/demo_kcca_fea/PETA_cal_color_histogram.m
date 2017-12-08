function hist_HS_RGB_Lab = PETA_cal_color_histogram(im_rgb, Weight)
    % Elyor, 05/02/2015, based on Kcca paper    
    
    % RGB
    nBins = 4;
    H_rgb=zeros([nBins nBins nBins]);
    for i=1:size(im_rgb, 1)
        for j=1:size(im_rgb, 2)
            p=double(reshape(im_rgb(i,j,:),[1 3]));
            p=floor(p/(256/nBins)) + 1;
            H_rgb(p(1),p(2),p(3))=H_rgb(p(1),p(2),p(3)) + Weight(i,j);
        end
    end
    H_rgb_fea_vec = reshape(H_rgb,[1 nBins^3]);
    H_rgb_fea_vec = NormalizeRows(H_rgb_fea_vec,'L2');
    
    %% HSV
    nBins = 8;
    im_hsv = rgb2hsv(im_rgb);
    H_hsv = zeros([nBins nBins]);
    for i=1:size(im_hsv, 1)
        for j=1:size(im_hsv, 2)
            p = double(reshape(im_hsv(i,j,1:2),[1 2]));
            p = floor(p/(1.01/nBins)) + 1;
            H_hsv(p(1),p(2))=H_hsv(p(1),p(2)) + Weight(i,j);
        end
    end
    H_hs_fea_vec = reshape(H_hsv,[1 nBins^2]);
    H_hs_fea_vec = NormalizeRows(H_hs_fea_vec,'L2');
    %% L*a*b (Not in the PAMI paper)
    H_Lab_fea_vec = [];
    nBins = 4;
    im_Lab = RGB2Lab(im_rgb);    % L -> [0 100]    % *a,*b -> [-110 110]
    H_Lab = zeros([nBins nBins nBins]);
    for i=1:size(im_Lab, 1)
        for j=1:size(im_Lab, 2)
            p(1) = double(im_Lab(i,j,1));
            p(2) = double(im_Lab(i,j,2));
            p(3) = double(im_Lab(i,j,3));
            p(1) = floor(p(1)/(101/nBins)) + 1;
            p(2) = p_lab(p(2));
            p(3) = p_lab(p(3));        
            H_Lab(p(1),p(2),p(3))=H_Lab(p(1),p(2),p(3)) + Weight(i,j);
        end
    end
    H_Lab_fea_vec = reshape(H_Lab,[1 nBins^3]);
    H_Lab_fea_vec = NormalizeRows(H_Lab_fea_vec,'L2');
    hist_HS_RGB_Lab = horzcat(H_rgb_fea_vec, H_hs_fea_vec, H_Lab_fea_vec);
end