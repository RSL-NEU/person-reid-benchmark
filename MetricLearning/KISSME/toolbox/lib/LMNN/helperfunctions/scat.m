function H=scat(Y,d,color,varargin);
% function scat(Y,d,color,varargin);
%
% displays points in Y with neigghbors from X
% within d dimensions 
% and color 
%
% Optional: pars.size   size of ball (40 default)
%           pars.circles=0 switches circles off
%           pars.cla=0   does not clear screen
%           pars.axeq=1  sets the axes to proper proportions
%           pars.nngraph   number of neighbors (default 0)
%		 width   line width
%
% example: 
% scat(X,3,color,'nngraph',3,'axeq',1,'cla',0);
%
% displays the columns of X as points, creates a k-NN graph with
% k=3, scales the axes but does not clear the figure beforehand
% 
% copyright Kilian Q. Weinberger 2006
% kilianw@seas.upenn.edu
  

pars.size=40;
pars.circles=1;
pars.axeq=1;
pars.cla=1;
pars.nngraph=0;
pars=extractpars(varargin,pars);

[D,N]=size(Y);

if(nargin<2) 
    d=2+(D>2);
end; 

if(d>3)
    d=3;
end;

oY=Y;
Y=Y(1:(min(d,size(Y,1))),:);
% fix the occlusion problem
if d>2
cp=get(gca,'CameraPosition')';

Yp=zeros(3,size(Y,2));
Yp(1:size(Y,1),:)=Y;
dd=distance(cp,Yp);
[temp,i]=sort(dd);
Y=Y(:,i);
oY=oY(:,i);
if(exist('color')==1)
 color=color(i);
end;
else
 i=1:size(Y,2);   
end;


if(pars.cla)
 cla;  
else
    hold on;
end;

 if(d>D) Y=[Y;zeros(d-D,N)]; end;
if(nargin>=3)
  if(d==2) H=scatter(Y(1,:),Y(2,:),pars.size,color,'filled');
   else H=scatter3(Y(1,:),Y(2,:),Y(3,:),pars.size,color,'filled');
  end;    
 if(pars.circles==1)
    set(H,'MarkerEdgeColor',[0.5 0.5 0.5]);
 end;
 hold on;
else 
 if(pars.circles==1 & nargin>3)
    set(H,'MarkerEdgeColor',[0.5 0.5 0.5]);
 else
  if(d==2) H=scatter(Y(1,:),Y(2,:),pars.size,'k','filled');
  else H=scatter3(Y(1,:),Y(2,:),Y(3,:),pars.size,'k','filled');
  end;
 end;
end;
if(pars.nngraph>0)
 W=createNNgraph(oY,pars.nngraph);
 if(d==3) gplot3(W,Y'); else gplot(W,Y');end;
end;
 hold on;

if(pars.axeq==1) axis equal;end;



function W=createNNgraph(x,k);
% function W=createNNgraph(x,k);
%
% outputs a sparse matrix with W_{ij}=1 if and only if 
% x(i) is amongst the k closest points of x_j)
%
% (Note: W is usually not symmetric)
%
% copyright Kilian Q. Weinberger 2005
%
  
  [d,n]=size(x);
  D=distance(x);
  [Ds,nn]=sort(D);
  nn=nn(2:k+1,:);
  
  W=sparse([],[],[],n,n);
  for i=1:n
   W(nn(:,i),i)=1;  
  end;
  

function pars=extractpars(vars,default);
% function pars=extractpars(vars,default);
%
%

if(nargin<2)
  default=[];
end;

pars=default;
if(length(vars)==1)
 p=vars{1};
 s=fieldnames(p);
 for i=1:length(s)
   eval(['pars.' s{i} '=p.' s{i}]);
 end;     

else




 for i=1:2:length(vars)
  if(isstr(vars{i}))
    if(i+1>length(vars)) error(sprintf('Parameter %s has no value\n',vars{i}));else val=vars{i+1};end;
    if(isstr(val))
     eval(['pars.' vars{i} '=''' val ''';']);
    else
     eval(['pars.' vars{i} '=val;']);
    end;
  end;
 end;

end;




function [Xout,Yout]=gplot3(A,xyz,lc)

[i,j] = find(A);
[ignore, p] = sort(max(i,j));
i = i(p);
j = j(p);


X = [ xyz(i,1) xyz(j,1) repmat(NaN,size(i))]';
Y = [ xyz(i,2) xyz(j,2) repmat(NaN,size(i))]';
Z = [ xyz(i,3) xyz(j,3) repmat(NaN,size(i))]';
X = X(:);
Y = Y(:);
Z = Z(:);

if nargout==0,
    if nargin<3,
        plot3(X, Y, Z)
    else
        plot3(X, Y, Z,lc);
    end
else
    Xout = X;
    Yout = Y;
end

