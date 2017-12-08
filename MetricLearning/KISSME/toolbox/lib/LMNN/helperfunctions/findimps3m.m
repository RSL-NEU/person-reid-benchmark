function ind=findimps3m(X1,X2,t);
% function ind=findimps3D(X1,X2,t1,t2);
%
% takes two input matrices X1,X2 ond one vector t
% 
% equivalent to: 
%     Dist=distance(X1.'*X2);
%     imp=find(Dist<repmat(t,N1,1))';
%     [a,b]=ind2sub([N1,N2],imp);
%
%
%

N1=size(X1,2);
N2=size(X2,2);
if(size(t,2)==1) t=t.';end;
Dist=distance2(X1,X2);  % computes L2-distance matrix
%imp=find(Dist<repmat(t,N1,1))';
imp=find(bsxfun(@lt,Dist,t))';
[a,b]=ind2sub([N1,N2],imp);

ind=[a;b];
ind=unique(ind','rows')';
end