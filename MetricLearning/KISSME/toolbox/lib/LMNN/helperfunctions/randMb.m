function Mb=randMB(d,n);
% function Mb=randMB(d,n);
%
% return a (d^2) x n  matrix  with each column being a vectorized
% symmetric random matrix
% 
%


Mb=rand(d^2,n);
for i=1:n
  M=rand(d);
  M=M+M.';
  Mb(:,i)=vec(M);
end;
