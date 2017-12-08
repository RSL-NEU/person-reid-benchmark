% Compute color (RGB, HSV, CbCr) and texture (responses of schmid and gabor
% filters) histograms
% im: input image, hist_n: number of stripes, hist_b: number of histogram
% bins

function x=colorTexture(I,nStripes,nBins)

x=[];
for i=1:numel(I)
    x(i,:)=colorTexture1(I{i},nStripes,nBins);
end

function x=colorTexture1(im,hist_n,hist_b)


feature_channel = 27;      %19 for texture, 8 for color, 27 in total

NF=13;                        % Number of filters
SUP=49;                       % Support of largest filter (must be odd)
F=zeros(SUP,SUP,NF);          % Schmit Filters init
fea_hist=zeros(feature_channel,hist_n*hist_b); % 27 channels in total

feature_dimension = hist_n*hist_b*feature_channel;

im=im2double(im);
img=rgb2gray(im);


%% Schmit Filters

F(:,:,1)=makeschfilter(SUP,2,1);
F(:,:,2)=makeschfilter(SUP,4,1);
F(:,:,3)=makeschfilter(SUP,4,2);
F(:,:,4)=makeschfilter(SUP,6,1);
F(:,:,5)=makeschfilter(SUP,6,2);
F(:,:,6)=makeschfilter(SUP,6,3);
F(:,:,7)=makeschfilter(SUP,8,1);
F(:,:,8)=makeschfilter(SUP,8,2);
F(:,:,9)=makeschfilter(SUP,8,3);
F(:,:,10)=makeschfilter(SUP,10,1);
F(:,:,11)=makeschfilter(SUP,10,2);
F(:,:,12)=makeschfilter(SUP,10,3);
F(:,:,13)=makeschfilter(SUP,10,4);


for i=1:13
    Imgabout = conv2(double(img),double(F(:,:,i)),'same');
    fea_hist(i,:)=tex_hist(Imgabout,hist_n,hist_b);
end

%% Gabor Filters

G1=gabor_fn(2^0.5,0,4,0,0.3);
G2=gabor_fn(2^0.5,0,8,0,0.3);
G3=gabor_fn(1^0.5,0,4,0,0.4);
G4=gabor_fn(2^0.5,pi/2,4,0,0.3);
G5=gabor_fn(2^0.5,pi/2,8,0,0.3);
G6=gabor_fn(1^0.5,pi/2,4,0,0.4);



for i=1:6
    eval(['Imgabout = conv2(double(img),double(G' num2str(i) '),''same'');']);
    fea_hist(13+i,:)=tex_hist(Imgabout,hist_n,hist_b);
end

n_ind=19;
% n_ind =0;
%% Color
fea_hist(n_ind+1,:)=rgb_hist(im(:,:,1),hist_n,hist_b);
fea_hist(n_ind+2,:)=rgb_hist(im(:,:,2),hist_n,hist_b);
fea_hist(n_ind+3,:)=rgb_hist(im(:,:,3),hist_n,hist_b);
im_hsv=rgb2hsv(im);
fea_hist(n_ind+4,:)=str_hist(im_hsv(:,:,1),hist_n,hist_b);
fea_hist(n_ind+5,:)=str_hist(im_hsv(:,:,2),hist_n,hist_b);
fea_hist(n_ind+6,:)=str_hist(im_hsv(:,:,3),hist_n,hist_b);
im_ybr=rgb2ycbcr(im);
fea_hist(n_ind+7,:)=str_hist(im_ybr(:,:,2),hist_n,hist_b);
fea_hist(n_ind+8,:)=str_hist(im_ybr(:,:,3),hist_n,hist_b);

% fea_hist(n_ind+1,:)=str_hist_gauss(im(:,:,1),hist_n,hist_b,sig);
% fea_hist(n_ind+2,:)=str_hist_gauss(im(:,:,2),hist_n,hist_b,sig);
% fea_hist(n_ind+3,:)=str_hist_gauss(im(:,:,3),hist_n,hist_b,sig);
% im_hsv=rgb2hsv(im);
% fea_hist(n_ind+4,:)=str_hist_gauss(im_hsv(:,:,1),hist_n,hist_b,sig);
% fea_hist(n_ind+5,:)=str_hist_gauss(im_hsv(:,:,2),hist_n,hist_b,sig);
% fea_hist(n_ind+6,:)=str_hist_gauss(im_hsv(:,:,3),hist_n,hist_b,sig);
% im_ybr=rgb2ycbcr(im);
% fea_hist(n_ind+7,:)=str_hist_gauss(im_ybr(:,:,2),hist_n,hist_b,sig);
% fea_hist(n_ind+8,:)=str_hist_gauss(im_ybr(:,:,3),hist_n,hist_b,sig);

[m,n]=size(fea_hist);


x=reshape(fea_hist',1,m*n);

function s_hist=rgb_hist(data,hist_n,hist_b)
% global POSE_PRIOR;
% global IMG_ANGLE;
% global PP_WEIGHT;

num_s=size(data,1);
num_c=size(data,2);
row_h=floor(num_s/hist_n);
s_hist=[];

%normalize
imdata=reshape(data,1,num_s*num_c);
mean_im=mean(imdata);
var_im=sqrt(var(imdata));

imdata=((imdata-mean_im)/var_im+2)/4;
imdata=reshape(imdata,num_s,num_c);

for i=1:hist_n
    data2=imdata((i-1)*row_h+1:i*row_h,:);
    %if(POSE_PRIOR == false || IMG_ANGLE==0)
        [hist_dat, hist_val]=imhist(data2,hist_b);
    %else
    %    [hist_dat, hist_val]=imhistW(data2, hist_b, PP_WEIGHT(i,:));
    %end
    hist_dat=hist_dat/sum(hist_dat);
    s_hist=[s_hist;hist_dat];
    
end

s_hist=reshape(s_hist',1,hist_n*hist_b);


function s_hist=str_hist(data,hist_n,hist_b)
% global POSE_PRIOR;
% global IMG_ANGLE;
% global PP_WEIGHT;

num_s=size(data,1);
num_c=size(data,2);

row_h=floor(num_s/hist_n);
s_hist=[];
for i=1:hist_n
    data2=data((i-1)*row_h+1:i*row_h,:);
    %if(POSE_PRIOR == false || IMG_ANGLE==0)
        [hist_dat, hist_val]=imhist(data2,hist_b);
    %else
    %    [hist_dat, hist_val]=imhistW(data2, hist_b, PP_WEIGHT(i,:));
    %end
    hist_dat=hist_dat/sum(hist_dat);
    s_hist=[s_hist;hist_dat];
    
end
s_hist=reshape(s_hist',1,hist_n*hist_b);

function s_hist=tex_hist(data,hist_n,hist_b)
% global POSE_PRIOR;
% global IMG_ANGLE;
% global PP_WEIGHT;

num_s=size(data,1);
num_c=size(data,2);
row_h=floor(num_s/hist_n);
s_hist=[];

imdata=reshape(data,1,num_s*num_c);
mean_im=mean(imdata);
std_im=sqrt(var(imdata));

imdata=((imdata-mean_im)/std_im+5)/10;
%imdata = (imdata+2)/4;
imdata=reshape(imdata,num_s,num_c);

for i=1:hist_n
    data2=imdata((i-1)*row_h+1:i*row_h,:);
    %if(POSE_PRIOR == false || IMG_ANGLE==0)
        [hist_dat, hist_val]=imhist(data2,hist_b);
    %else
    %    [hist_dat, hist_val]=imhistW(data2, hist_b, PP_WEIGHT(i,:));
    %end
    hist_dat=hist_dat/sum(hist_dat);
    s_hist=[s_hist;hist_dat];
end
s_hist=reshape(s_hist',1,hist_n*hist_b);

function f=makeschfilter(sup,sigma,tau)
  hsup=(sup-1)/2;
  [x,y]=meshgrid(-hsup:hsup);
  r=(x.*x+y.*y).^0.5;
  f=cos(r*(pi*tau/sigma)).*exp(-(r.*r)/(2*sigma*sigma));
  f=f-mean(f(:));      
  f=f/sum(abs(f(:)));    

function gb=gabor_fn(sigma,theta,lambda,psi,gamma)
 
sigma_x = sigma;
sigma_y = sigma/gamma;
 
% Bounding box
nstds = 3;
xmax = max(abs(nstds*sigma_x*cos(theta)),abs(nstds*sigma_y*sin(theta)));
xmax = ceil(max(1,xmax));
ymax = max(abs(nstds*sigma_x*sin(theta)),abs(nstds*sigma_y*cos(theta)));
ymax = ceil(max(1,ymax));
xmin = -xmax; ymin = -ymax;
[x,y] = meshgrid(xmin:xmax,ymin:ymax);
 
% Rotation 
x_theta=x*cos(theta)+y*sin(theta);
y_theta=-x*sin(theta)+y*cos(theta);
 
gb= exp(-.5*(x_theta.^2/sigma_x^2+y_theta.^2/sigma_y^2)).*cos(2*pi/lambda*x_theta+psi);
%gb = gb/sum(abs(gb(:)));

