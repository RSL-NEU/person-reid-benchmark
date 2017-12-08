function dist =MatchScore(X1, X2,Dict,par)
%% function: Compute the matching response value between X1 and X2
%% input:
%   X1 : D x N matrix
%   X2 : D x M matrix
%   par : parameters 
%% output:
%  dist : M x N matrix

[dim N]=size(X1);
M=size(X2,2);
dist=zeros(M,N);

Dx = Dict.Dx;
Dw = Dict.Dw;
Mx = Dict.Mx;
Mw = Dict.Mw;

Alphaf=inv(Dx'*Dx+par.lambdac*eye(size(Dx,2)))* Dx'*X1;
Alphaw = Mx * Alphaf;
W = Dw * Alphaw;

for i=1:N
    Diff=[repmat(X1(:,i),1,M);abs(repmat(X1(:,i),1,M)-X2);X2];  
    Response=Diff'*W(:,i); 
    dist(:,i)=Response;
end

end