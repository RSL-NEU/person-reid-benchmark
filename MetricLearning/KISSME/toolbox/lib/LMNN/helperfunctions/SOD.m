function res=SOD(x,a,b)
% function res=SOD(x,a,b);
%
% Computes and sums over all outer products of the columns in x. 
%
% equivalent to:
%
% res=zeros(size(x,1));
% for i=1:n
%   res=res+x(:,a(i))*x(:,b(i))';
% end;
%
%
% copyright 2005 by Kilian Q. Weinberger
% University of Pennsylvania
% kilianw@seas.upenn.edu
% ********************************************
%  Xa=x(:,a);
%  Xb=x(:,b);
%  XaXb=Xa*Xb';
%  res=Xa*Xa'+Xb*Xb'-XaXb-XaXb';


[D,~]=size(x);
B=ceil(134217728/D); % max 1MB of memory per block
res=zeros(D);
for i=1:B:length(a)
  BB=min(B-1,length(a)-i);
  Xa=x(:,a(i:i+BB));
  Xb=x(:,b(i:i+BB));
  XaXb=Xa*Xb';
  res=res+Xa*Xa'+Xb*Xb'-XaXb-XaXb'; 
  if(i>1) fprintf('.');end;
end;
%res=reshape(res,[D D]);  
  

  
