function [Eval,Details,tree]=knnclassifytree(L,xTr,lTr,xTe,lTe,KK,varargin)
% function [Eval,Details]=knnclassifytree	(tree,xTr,lTr,xTe,lTe,KK,varargin);
%
% INPUT:
%  L    :   linear transformation
%  xTr	:   training vectors (each column is an instance)
%  yTr	:   training labels  (row vector!!)
%  xTe  :   test vectors
%  yTe  :   test labels
%  Kg	:   number of nearest neighbors
%
% OUTPUT:
% 
%
% Optional Input:
% 'tree',tree (precomputed,mtree)
% 'teesize',15 (max number of elements in leaf)
% 'train',1  (0 means no training error)
% 
% Good luck!
%
% copyright by Kilian Q. Weinberger, 2006
%  
% version 1.1  (04/13/07) 
% Little bugfix, couldn't handle single test vectors beforehand. 
% Thanks to Karim T. Abou-Moustafa for pointing it out to me. 
%


pars.train=1;
pars.tree=[];
pars.treesize=15;
pars=extractpars(varargin,pars);


if(~isempty(L))
 dim=size(L,2);
 xTr=L*xTr(1:dim,:);
 xTe=L*xTe(1:dim,:); 
else
 dim=size(xTr,1);    
end;

if(isempty(pars.tree))
    fprintf('Building tree ...');
    pars.tree=buildmtreemex(xTr,pars.treesize);
    fprintf('done\n');
end;

MM=min([lTr lTe]);
if(nargin<6)
 KK=1:2:3;
end;
Kn=max(KK);

[NTr]=size(xTr,2);
[NTe]=size(xTe,2);

if(size(xTr,1)~=size(xTe,1))
 fprintf('PROBLEM: Please make sure that training inputs and test inputs have the same dimensions!\n');
 fprintf('size(xTr)');
 size(xTr)
 fprintf('size(xTe)');
 size(xTe)
 Eval=[];
 Details=[];
 return;
end;



if(max(max(pars.tree.jumpindex))~=NTr)
 fprintf('PROBLEM: Tree does not seem to belong to training data!\n');
 fprintf('Max index of tree: %i\n',max(max(pars.tree.jumpindex)));
 fprintf('Length of training data: %i\n',NTr);
 Eval=[];
 Details=[];
 return;
end;

if(length(KK)==1) outputK=KK;KK=1:KK;else outputK=1:length(KK);end;



Eval=zeros(2,length(KK));

[iTe,dists]=usemtreemex(xTe,xTr,pars.tree,Kn);
lTe2=LSKnn2(reshape(lTr(iTe),max(KK),NTe),KK,MM);
Eval(2,:)=sum((lTe2~=repmat(lTe,length(KK),1))',1)./NTe;


if(pars.train)
 [iTr,dists]=usemtreemex(xTr,xTr,pars.tree,Kn+1);
 iTr=iTr(2:end,:);
 lTr2=LSKnn2(lTr(iTr),KK,MM);
 Eval(1,:)=sum((lTr2~=repmat(lTr,length(KK),1))',1)./NTr;

 Details.lTr2=lTr2;
 Details.iTr=iTr;
else
 Eval(1,:)=[]; 
end;

Details.lTe2=lTe2;
Details.iTe=iTe;
Eval=Eval(:,outputK);

disp(Eval.*100)

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

