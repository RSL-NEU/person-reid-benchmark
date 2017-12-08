function [Eval,Details]=knnE(Lb,xTr,lTr,xTe,lTe,KK,varargin)
% function [Eval,Details]=knnE(Lb,xTr,lTr,xTe,lTe,KK);
%

pars.train=1;
pars.test=1;
pars.verbose=1;
pars=extractpars(varargin,pars);

MM=min([lTr lTe]);
if(nargin<6)
 KK=1:2:3;
end;

if(length(KK)==1) outputK=ceil(KK/2);KK=1:2:KK;else outputK=1:length(KK);end;

Kn=max(KK);

B=700;
[NTr]=size(xTr,2);
[NTe]=size(xTe,2);

Eval=zeros(2,length(KK));
lTr2=zeros(length(KK),NTr);
lTe2=zeros(length(KK),NTe);

iTr=zeros(Kn,NTr);
iTe=zeros(Kn,NTe);


sx1=sum(xTr.^2);
sx2=sum(xTe.^2);
NN=max(NTr*pars.train,NTe*pars.test);

for i=1:B:NN
  if(i<NTr && pars.train)
   BTr=min(B-1,NTr-i);  
   Dtr=distanceE(Lb,xTr,xTr(:,i:i+BTr));   
   [dist,nn]=mink(Dtr,Kn+1);
   nn=nn(2:Kn+1,:);
   lTr2(:,i:i+BTr)=LSKnn2(lTr(nn),KK,MM);
   iTr(:,i:i+BTr)=nn;
   Eval(1,:)=sum((lTr2(:,1:i+BTr)~=repmat(lTr(1:i+BTr),length(KK),1))')./(i+BTr);
  end;
  if(i<NTe && pars.test)
   BTe=min(B-1,NTe-i);  
   Dtr=distanceE(Lb,xTr,xTe(:,i:i+BTe));
   [dist,nn]=mink(Dtr,Kn);
   lTe2(:,i:i+BTe)=LSKnn2(lTr(nn),KK,MM);
   iTe(:,i:i+BTe)=nn;   
   Eval(2,:)=sum((lTe2(:,1:i+BTe)~=repmat(lTe(1:i+BTe),length(KK),1))')./(i+BTe);
  end;   
  if(pars.verbose)
   	progressbar(i+B,NN,[' Test error:' num2str(Eval(end).*100) '% ']); 
  end;
end;

if(pars.verbose)
	fprintf('\n');
end;
Details.lTe2=lTe2;
Details.lTr2=lTr2;
Details.iTe=iTe;
Details.iTr=iTr;	
Eval=Eval(:,outputK);






function yy=LSKnn2(Ni,KK,MM);
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
  T(i,:,k)=sum(Ni(1:k,:)==c,1);
 end;
end;
yy=zeros(max(KK),N);
for k=KK
 [temp,yy(k,1:N)]=max(T(:,:,k)+T(:,:,1).*0.01);
 yy(k,1:N)=classes(yy(k,:));
end;
yy=yy(KK,:);
yy=yy+MM-1;








function D=distanceE(Lb,x2,x1);
% computes the distance for points in clusters with their own
% Mahalanobis metric
% copyright 02/2006 by Kilian Q. Weinberger
% kilianw@seas.upenn.edu
  
D=zeros(size(x2,2),size(x1,2));
[Lm,Ln]=size(Lb.L{1});
for i=1:length(Lb.un)
 j=find(Lb.E==Lb.un(i));
 L=Lb.L{i};
 D(j,:)=distance(L*x2(1:Ln,j),L*x1(1:Ln,:));
end;

