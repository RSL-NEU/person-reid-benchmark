function C=SODWkqwAx(X,W)
% function C=SODWkqw(X,W)
%
% 
% Returns the sum of all weighted outer products
%
% C=A*\sum_{i,j} (X(:,i)-X(:,j))(X(:,i)-X(:,j))'*W(i,j)


Q=W+W';
ii=find(sum(Q~=0));
Q=Q(ii,ii);
X=X(:,ii);
C=-(X*(Q-spdiags(sum(Q)',0,length(Q),length(Q))))*X';


%Q=Q-spdiags(sum(Q)',0,length(Q),length(Q));toc
%C=(bsxfun(@times,AX,full(sum(Q))))*X'-AX*Q*X';
%s=full(sum(Q));
%C=(AX.*(ones(size(AX,1),1)*s)*X')-AX*Q*X';




