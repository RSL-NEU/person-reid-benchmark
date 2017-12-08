function GaborFeature = gaborextractionmultiscale(ImgData,ImgSize,m_nScale,m_nOrientation)
% Input: 
%       ImgData:  the image data, a vector with size of [hei¡Áwid£¬1]
%       ImgSize: the size of image£¬width = ImgSize(1);height = ImgSize(2);
%       m_nScale: the number of Gabor scales
%       m_nOrientation: the number of Gabor orientation. 
% Output:
%       GaborFeature
% Notes:
%       We apply MAX pooling over two consecutive scales (within the
%       same orientation)

% * current version£º1.0
% * Author£ºBingpeng MA
% * Date£º2009-12-21

dim = ImgSize(1) * ImgSize(2);
GaborFeature = zeros(dim, m_nScale/2 * m_nOrientation);
for cross_scale = 1 : m_nScale/2
    filtersize = cross_scale * 4 - 1;
    [GaborReal,GaborConj]= kernelcreate1(filtersize,filtersize,cross_scale,m_nOrientation,2*pi,2*pi,0.7888*pi,sqrt(2));
    Gabor = GaborReal  + 1i * GaborConj;
    GaborFeature_odd  = GaborExtraction1(ImgData, ImgSize, Gabor,filtersize,filtersize);
    filtersize = cross_scale *4 + 1;
    [GaborReal,GaborConj]= kernelcreate1(filtersize,filtersize,cross_scale,m_nOrientation,2*pi,2*pi,0.7888*pi,sqrt(2));
    Gabor = GaborReal  + 1i * GaborConj;
    GaborFeature_even  = GaborExtraction1(ImgData, ImgSize, Gabor,filtersize,filtersize);
    GaborFeature_temp = max(GaborFeature_even,GaborFeature_odd);
    clear GaborFeature_even GaborFeature_odd GaborReal GaborConj Gabor;
    index_begin = (cross_scale-1)* m_nOrientation  +1;
    index_end = cross_scale * m_nOrientation;
    GaborFeature(:,index_begin:index_end) = reshape(GaborFeature_temp,dim,size(GaborFeature_temp,1)/dim);
    clear GaborFeature_temp;
end
return;