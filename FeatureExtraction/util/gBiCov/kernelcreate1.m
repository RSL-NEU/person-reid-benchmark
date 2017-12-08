function  [GaborReal  GaborConj]= kernelcreate1(KernelWidth,KernelHeight, scale, nOrientation,  xSigma,   ySigma,  Kmax,   Frequency )
% Creat the kernel of Gabor filters
%notes:
%pImagereal = KernelCreate(32,32,5,8,2*pi,2*pi,pi,sqrt(2));

m_nScale = scale;
m_nOrientation = nOrientation;
m_dSigmaY = xSigma ;
m_dSigmaX = ySigma ;
m_dfrequency = Frequency;
m_dKmax = Kmax;

m_nGaborWinHeight = KernelHeight;
m_nGaborWinWidth  = KernelWidth;
m_bParamentsSet = true ;

if(m_bParamentsSet == false)
    m_dSigma  = 2*pi;
    m_dfrequency = sqrt(2);
    m_dKmax  = pi/2;
else
    m_dSigma = m_dSigmaX;
end

postConstant  = exp(-m_dSigma*m_dSigma/2);
[widthup,widthdown,heightdown, heightup] = SetWinodwsSize(m_nGaborWinWidth,m_nGaborWinHeight);%,&widthup,&widthdown,&heightdown,&heightup);
newsize = m_nGaborWinWidth * m_nGaborWinHeight;
GaborReal = zeros(newsize,1*m_nOrientation);
GaborConj = zeros(newsize,1*m_nOrientation);

cross_Gabor = 0 ;
i=  scale-1;
x = heightdown :heightup;
x = repmat(x',1, widthup-widthdown+1);
y = repmat(widthdown: widthup, heightup-heightdown+1,1);
xy = (x.*x/(m_dSigmaX*m_dSigmaX) + y.*y/(m_dSigmaY*m_dSigmaY));
for j=0:m_nOrientation-1
    Phi = j * pi / m_nOrientation;
    cross_Gabor = cross_Gabor +1;
    preConstant_Kuv = (m_dKmax / m_dfrequency.^i).^2.0;
    Kv  = m_dKmax / m_dfrequency.^i;
    exppart = exp(-1 * preConstant_Kuv * xy) * preConstant_Kuv / (m_dSigmaX*m_dSigmaY) ;
    GaborReal_temp  = exppart .* ( cos(  Kv*(  cos(Phi)*x + sin(Phi)*y ) ) - postConstant );
    GaborConj_temp =  exppart .* ( sin(  Kv*(  cos(Phi)*x + sin(Phi)*y ) ) );
    GaborReal(:, cross_Gabor)  = GaborReal_temp(:);
    GaborConj(:, cross_Gabor) =  GaborConj_temp(:);
end
return;


function [widthup,widthdown,heightdown, heightup] = SetWinodwsSize(imagewidth,imageheight)

remainder = mod(imagewidth , 2);
if(remainder == 0 )
    widthdown = -(imagewidth / 2 - 1);
    widthup   = imagewidth / 2 ;
else
    widthdown = - (imagewidth-1)/ 2;
    widthup   =  (imagewidth-1) / 2;
end

remainder = mod(imageheight,2);
if(remainder == 0 )
    heightdown = -(imageheight / 2 - 1);
    heightup   = imageheight / 2 ;
else
    heightdown = - (imageheight-1)/ 2;
    heightup   =  (imageheight-1)/ 2;
end

return;