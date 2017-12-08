function [err,yy,Value]=energyclassify(Lb,x,y,xTest,yTest,Kg,varargin);
% function [err,yy,Value]=energyclassify(L,xTr,yTr,xTe,yTe,Kg,varargin);
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



% set parameters
pars.alpha=1e-09;
pars.tempid=0;
pars.save=0;
pars.speed=10;
pars.skip=0;
pars.factor=1;
pars.correction=15;
pars.prod=0;
pars.thresh=1e-16;
pars.ifraction=1;
pars.scale=0;
pars.obj=0;
pars.union=1;
pars.tabularasa=Inf;
pars.blocksize=500;
pars.margin=0;
pars.v2w=1;
pars.t1=1;
pars.t2=1;
pars.t3=1;
pars=extractpars(varargin,pars);



% make sure to read Lb correctly even if rectangular
[Lm,Ln]=size(Lb.L{1});
x=x(1:Ln,:);
xTest=xTest(1:Ln,:);

tempname=sprintf('temp%i.mat',pars.tempid);
% Initializationip
[D,N]=size(x);
[gen,NN]=getGenLS(x,y,Kg,pars);



if(pars.scale)
 fprintf('Scaling input vectors!\n');
 sc=sqrt(mean(sum( ((x-x(:,NN(end,:)))).^2)));
 x=x./sc;
 xTest=xTest./sc;
end;



for i=1:length(Lb.un)
 L=Lb.L{i};
 Lx{i}=L*x;
 Lx2{i}=sum(Lx{i}.^2);
 LxT{i}=L*xTest;
end;
clear('L');

un=Lb.un;
for c=1:length(Lx)
 for inn=1:Kg
  j=find(y(NN(inn,:))==un(c));
  Ni(inn,j)=(sum((Lx{c}(:,j)-Lx{c}(:,NN(inn,j))).^2)+1);
 end;
end;


un=unique(y);
Value=zeros(length(un),length(yTest));
t1=zeros(length(un),length(yTest));
t2=zeros(length(un),length(yTest));
t3=zeros(length(un),length(yTest));

B=pars.blocksize;
if(size(x,2)>50000) B=250;end;
NTe=size(xTest,2);
for n=1:B:NTe
  fprintf('%2.2f%%: ',n/NTe*100);
  nn=n:n+min(B-1,NTe-n);

 % distance with OTHER's metric
  DDo=zeros(size(Lx{1},2),length(nn));
  for i=1:length(un)
   jj=find(y==un(i));
   DDo(jj,:)=distance(Lx{i}(:,jj),LxT{i}(:,nn));  
  end;

 for i=1:length(un)
  % distance with MY metric
   DDm=distance(Lx{i},LxT{i}(:,nn));   
  testlabel=un(i);
  fprintf('%i.',testlabel);
  
  enemy=find(y~=testlabel);
  friend=find(y==testlabel);

  Df=mink(DDo(friend,:),Kg);
  Value(i,nn)=sum(Df,1)+sumiflessh2(DDo,Df+pars.margin,enemy);
  v2(i,nn)=sumiflessv2(DDm,Ni(:,enemy),enemy);

  Df=mink(DDo(friend,:),Kg);
%  Value(i,nn)=sumiflessv2(DDm,Ni(:,enemy),enemy)+sumiflessh2(DDo,Df,enemy)+sum(Df,1);
  t1(i,nn)=sumiflessv2(DDm,Ni(:,enemy),enemy);
  t2(i,nn)=sumiflessh2(DDo,Df,enemy);
  t3(i,nn)=sum(Df,1);

 end;
 Value=pars.t1.*t1+pars.t2.*t2+pars.t3.*t3;

 % compute error up to now
 [temp,yy]=min(Value(:,1:max(nn)));
 yy=un(yy);
 err=sum(yy~=yTest(1:max(nn)))./length(yTest(1:max(nn)));
 fprintf(' err:%2.2f%%\n',err*100);
end;

fprintf('\n');
fprintf('Energy error:%2.2f%%\n',err*100);






function [gen,NN]=getGenLS(x,y,Kg,pars);
fprintf('Computing nearest neighbors ...\n');
[D,N]=size(x);
if(pars.skip) load('.LSKGnn.mat');
else
un=unique(y);
Gnn=zeros(Kg,N);
for c=un
fprintf('%i nearest genuine neighbors for class %i:',Kg,c);
i=find(y==c);
nn=LSKnn(x(:,i),x(:,i),2:Kg+1);
Gnn(:,i)=i(nn);
fprintf('\r         ');
end;

end;
NN=Gnn;
gen1=vec(Gnn(1:Kg,:)')';
gen2=vec(repmat(1:N,Kg,1)')';

gen=[gen1;gen2];

if(pars.save)
save('.LSKGnn.mat','Gnn');
end; 



function NN=LSKnn(X1,X2,ks,pars)
B=2000;
[D,N]=size(X2);
NN=zeros(length(ks),N);
DD=zeros(length(ks),N);

for i=1:B:N
  BB=min(B,N-i);
  fprintf('.');
  Dist=distance(X1,X2(:,i:i+BB));
  fprintf('.');
%  [dist,nn]=sort(Dist);
  [dist,nn]=mink(Dist,max(ks));
  clear('Dist');
  fprintf('.'); 
%  keyboard;
  NN(:,i:i+BB)=nn(ks,:);
  clear('nn','dist');
  fprintf('(%i%%) ',round((i+BB)/N*100)); 
end;


function I=updateOuterProduct(x,vals,active1,active2,a1,a2);



function v=vec(M);
% vectorizes a matrix

v=M(:);

