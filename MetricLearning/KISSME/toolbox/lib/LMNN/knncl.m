function [Eval,Details]=knncl(L,xTr,lTr,xTe,lTe,KK,varargin)
% function [Eval,Details]=knncl(L,xTr,yTr,xTe,yTe,Kg);
%
% INPUT:
%  L	:   transformation matrix (learned by LMNN)
%  xTr	:   training vectors (each column is an instance)
%  yTr	:   training labels  (row vector!!)
%  xTe  :   test vectors
%  yTe  :   test labels
%  Kg	:   number of nearest neighbors
%
% Good luck!
%
% copyright by Kilian Q. Weinberger, 2006
%  
% version 1.2 (10/05/11) small speedups for multi-core cpus
% version 1.1  (04/13/07) 
% Little bugfix, couldn't handle single test vectors beforehand. 
% Thanks to Karim T. Abou-Moustafa for pointing it out to me. 
%

pars.train=1;
pars.test=1;
pars.cosigndist=0;
pars.blocksize=700;
pars=extractpars(varargin,pars);

MM=min([lTr lTe]);
if(nargin<6)
 KK=1:2:3;
end;

if(length(KK)==1) outputK=ceil(KK/2);KK=1:2:KK;else outputK=1:length(KK);end;

Kn=max(KK);

if(~isempty(L))
 D=length(L);
 xTr=L*xTr(1:D,:);
 if(~isempty(xTe))
   xTe=L*xTe(1:D,:);
 end;
else
 D=size(xTr,1);
end;

[NTr]=size(xTr,2);
[NTe]=size(xTe,2);

Eval=zeros(2,length(KK));
lTr2=zeros(length(KK),NTr);
lTe2=zeros(length(KK),NTe);

iTr=zeros(Kn,NTr);
iTe=zeros(Kn,NTe);


sx1=sum(xTr.^2,1);
sx2=sum(xTe.^2,1);

if(~pars.train) 
    NTr=0;
end;

for i=1:pars.blocksize:max(NTr,NTe)
  if(pars.train && i<=NTr)
   	BTr=min(pars.blocksize-1,NTr-i);  
	% Dtr=distance(xTr,xTr(:,i:i+BTr));
   Dtr = bsxfun(@plus,sx1.',bsxfun(@plus,sx1(i:i+BTr),-2*xTr.'*xTr(:,i:i+BTr)));
	 
    [~,nn]=mink(Dtr,Kn+1);
   	nn=nn(2:Kn+1,:);
   	lTr2(:,i:i+BTr)=LSKnn2(lTr(nn),KK,MM);
   	iTr(:,i:i+BTr)=nn;
   	Eval(1,:)=sum((lTr2(:,1:i+BTr)~=repmat(lTr(1:i+BTr),length(KK),1))',1)./(i+BTr);
  end;
  if(pars.test && i<=NTe)
   	BTe=min(pars.blocksize-1,NTe-i);  
  	% Dtr=distance(xTr,xTe(:,i:i+BTe));
    Dtr = bsxfun(@plus,sx1.',bsxfun(@plus,sx2(i:i+BTe),-2*xTr.'*xTe(:,i:i+BTe)));
   	[~,nn]=mink(Dtr,Kn);
   	lTe2(:,i:i+BTe)=LSKnn2(reshape(lTr(nn),max(KK),BTe+1),KK,MM);
   	iTe(:,i:i+BTe)=nn;   
   	Eval(2,:)=sum((lTe2(:,1:i+BTe)~=repmat(lTe(1:i+BTe),length(KK),1))',1)./(i+BTe);
  end;   
  if(pars.train && pars.test)
   progressbar(i+BTr,max(NTr,NTe),sprintf('%2.2f/%2.2f ',100.*Eval(:)));
  end;
  if(pars.test && ~pars.train)
   progressbar(i+BTe,NTe,sprintf('%2.2f/%2.2f ',100.*Eval(:)));
  end;
  if(~pars.test && pars.train)
   progressbar(i+BTr,NTr,sprintf('%2.2f/%2.2f ',100.*Eval(:)));
  end;
end;


% create "Details" output
if(pars.test)
 Details.lTe2=lTe2;
 Details.iTe=iTe;
end;
if(pars.train)
 Details.lTr2=lTr2;
 Details.iTr=iTr;
end;

% extract "Eval" output
if(pars.train && pars.test)
 Eval=Eval(:,outputK);
end; 
if(pars.train && ~pars.test)
 Eval=Eval(1,outputK);
end;
if(~pars.train && pars.test)
 Eval=Eval(2,outputK);
end;

function yy=LSKnn2(Ni,KK,MM)
% function yy=LSKnn2(Ni,KK,MM);
%

if(nargin<2)
 KK=1:2:3;
end;

N=size(Ni,2);
Ni=Ni-MM+1;
classes=unique(unique(Ni))';



T=zeros(length(classes),N,length(KK));


for i=1:length(classes)
c=classes(i);  
 for k=KK
try
  T(i,:,k)=sum(Ni(1:k,:)==c,1);
catch
keyboard;
end;
 end;
end;

yy=zeros(max(KK),N);
for k=KK
 [temp,yy(k,1:N)]=max(T(:,:,k)+T(:,:,1).*0.01);
 yy(k,1:N)=classes(yy(k,:));
end;
yy=yy(KK,:);

yy=yy+MM-1;




