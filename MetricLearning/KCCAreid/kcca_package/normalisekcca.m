function [re] = normalise(A,K)

%
% Normalise features, used withint various KCCA code
%
% David R. Hardoon - drh@ecs.soton.ac.uk
%

%n = size(A,2);
n = size(A,1);
KK = K'*K;

%for i=1:n
%  comp = sqrt(A(:,i)'*KK*A(:,i));
%  if comp ~= 0
%    re(:,i) = A(:,i)/comp;
%  else
%    re(:,i) = A(:,i);
%  end
%end

comp = sqrt(diag(A'*KK*A));
comp = comp+(comp==0);

re = A./(ones(n,1)*comp');
