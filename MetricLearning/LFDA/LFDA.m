function [T,Z]=LFDA(X,Y,r,metric,kNN,reg)
%
% Local Fisher Discriminant Analysis for Supervised Dimensionality Reduction
%
% Usage:
%       [T,Z]=LFDA(X,Y,r,metric)
%
% Input:
%    X:      d x n matrix of original samples
%            d --- dimensionality of original samples
%            n --- the number of samples 
%    Y:      n dimensional vector of class labels
%    r:      dimensionality of reduced space (default: d)
%    metric: type of metric in the embedding space (default: 'weighted')
%            'weighted'        --- weighted eigenvectors 
%            'orthonormalized' --- orthonormalized
%            'plain'           --- raw eigenvectors
%    kNN:    parameter used in local scaling method (default: 7)
%
% Output:
%    T: d x r transformation matrix (Z=T'*X)
%    Z: r x n matrix of dimensionality reduced samples 
%
% (c) Masashi Sugiyama, Department of Compter Science, Tokyo Institute of Technology, Japan.
%     sugi@cs.titech.ac.jp,     http://sugiyama-www.cs.titech.ac.jp/~sugi/software/LFDA/

if nargin<2
  error('Not enough input arguments.')
end
[d n]=size(X);

if nargin<3
  r=d;
end

if nargin<4
  metric='weighted';
end

if nargin<5
  kNN=7;
end

tSb=zeros(d,d);
tSw=zeros(d,d);

for c=unique(Y')
  Xc=X(:,Y==c);
  nc=size(Xc,2);
  %if(fda)
  %    A=ones(nc,nc);
  %else
      % Define classwise affinity matrix
      Xc2=sum(Xc.^2,1);
      distance2=repmat(Xc2,nc,1)+repmat(Xc2',1,nc)-2*Xc'*Xc;
      [sorted,index]=sort(distance2);
      %kNNdist2=sorted(kNN+1,:);
      kNNdist2=sorted(min(kNN+1,nc),:);
      sigma=sqrt(kNNdist2);
      localscale=sigma'*sigma;
      flag=(localscale~=0);
      A=zeros(nc,nc);
      A(flag)=exp(-distance2(flag)./localscale(flag));
  %end
  Xc1=sum(Xc,2);
  G=Xc*(repmat(sum(A,2),[1 d]).*Xc')-Xc*A*Xc';
  tSb=tSb+G/n+Xc*Xc'*(1-nc/n)+Xc1*Xc1'/n;
  tSw=tSw+G/nc;
end

X1=sum(X,2);
tSb=tSb-X1*X1'/n-tSw;

tSb=(tSb+tSb')/2;
tSw=(tSw+tSw')/2;

tSw = reg * tSw + (1-reg)*trace(tSw)/n * eye(d);

if r==d
  [eigvec,eigval_matrix]=eig(tSb,tSw);
else
  opts.disp = 0; 
  [eigvec,eigval_matrix]=eigs(tSb,tSw,r,'lr',opts);
end
eigval=diag(eigval_matrix);
[sort_eigval,sort_eigval_index]=sort(eigval);
T0=eigvec(:,sort_eigval_index(end:-1:1));

switch metric %determine the metric in the embedding space
  case 'weighted'
   T=T0.*repmat(sqrt(sort_eigval(end:-1:1))',[d,1]);
  case 'orthonormalized'
   [T,dummy]=qr(T0,0);
  case 'plain'
   T=T0;
end

Z=T'*X;

