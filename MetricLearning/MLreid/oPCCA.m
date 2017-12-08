% PCCA distance learning algorithm proposed in "PCCA: A New Approach for
% Distance Learning from sparse pairwise constraints" cvpr 2012
%
% By Mengran Gou @ 02/24/2016
function [Method, l_old]= oPCCA(X, ix_pair, y, option)
ix_pair= double(ix_pair);
A =[];
l_old = [];
X = single(X);
% AKA = [];
display(['begin PCCA ...']);
beta= option.beta;
d = option.d;
eps = option.epsilon;
eta = 0.1;
max_iter = 1000;

diff = X(ix_pair(:,1),:) - X(ix_pair(:,2),:);
%%
%initialization
[v,w] = eigs(double(cov(X)),d,'la');
A = bsxfun(@times,v,1./sqrt(diag(w))');
A = A';
A = A(end:-1:1,:);
% A =randn(d, size(X,1))/1e4;
% temp = K(ix_pair(:,1),:) - K(ix_pair(:,2),:);
projdiff = A*diff';
% temp = temp*A';
A = A./mean(sqrt(sum(projdiff.^2,1))); %initialize with whitening PCA

projdiff = A*diff';
D = sum(projdiff.^2, 1);
% AKA =trace(A*X*A');
l_old = sum(logistic_loss(D,y,option)); %

% gradient search
cnt =0;
while 1
    L = -y'.*(1 -D); % there is a sign mistake in equation (3) and equation for L_n^t
    L = 1./ (1+exp(-beta*L)); 
    L = double(y'.*L); %d_loss
    Y = bsxfun(@times,projdiff,L);
    Y = Y*diff;
%     if option.lambda ==0
%         Y = bsxfun(@times,projdiff,L);
%         Y = Y*Kj;
% %         Y = reshape(L*KJ,[size(X,1) size(X,1)]);
%     else
%         Y = bsxfun(@times,projdiff,L);
%         Y = Y*Kj+option.lambda*A;
% %         Y = (diff*A')*L*Kj + W;
% %         Y = reshape(L*KJ,[size(X,1) size(X,1)]) + W;
%     end
%     % optimization eta does not work
%     temp =reshape(sum(KJ),[size(X,1) size(X,1)]);
%     temp = temp*K*Y*A'*A;
%     eta = trace(temp)/(2*trace(temp*Y));
    
%     A_new = A - 2*A *eta *Y;
    A_new = A - 2*eta*Y;
%     diff = K(ix_pair(:,1),:) - K(ix_pair(:,2),:);
    projdiff = A_new*diff';
    D = sum(projdiff.^2, 1);
%     AKA_new =trace(A_new*K*A_new');
    l_new = sum(logistic_loss(D,y,option));
    % adjust learning rate
    while l_new >  l_old
        eta = eta*0.9;
        if eta <1e-50
            break;
        else
            A_new = A - 2*eta*Y;
        %     diff = K(ix_pair(:,1),:) - K(ix_pair(:,2),:);
            projdiff = A_new*diff';
            D = sum(projdiff.^2, 1);
%             AKA_new =trace(A_new*K*A_new');
            l_new = sum(logistic_loss(D,y,option));
        end
    end
%     if l_new >  l_old
%         eta = eta*0.9;
%         if eta <1e-50
%             break;
%         else
%             continue;
%         end
%     else
        L = -y'.*(1 -D); 
        L = 1./ (1+exp(-beta*L));
        L = double(y'.*L);
        Y = bsxfun(@times,projdiff,L);
        Y = Y*diff;
%         if option.lambda ==0
%             Y = bsxfun(@times,projdiff,L);
%             Y = Y*Kj;
%         else
% %             Y = reshape(L*KJ,[size(X,1) size(X,1)]) + W;
%             Y = bsxfun(@times,projdiff,L);
%             Y = Y*Kj+option.lambda*A_new;
%         end
        df = 2*Y;
        df_norm = sqrt(sum(dot(df,df)));
        eta = (l_old-l_new)/df_norm;
%         eta = eta*1.1;
%     end
    if mod(cnt,100)==0
        display(num2str([ cnt l_new  norm(A-A_new,'fro')/norm(A, 'fro') eta ]))
%         plot(D); drawnow; pause(0.1)
    end
    
    if l_old - l_new < eps && norm(A-A_new, 'fro')/norm(A, 'fro')<eps
        break;
    end
    l_old = l_new;
    A = A_new;
%     AKA = AKA_new;
    cnt =cnt+1;
    if cnt > max_iter
        break;
    end
end
display(num2str([ cnt l_old  norm(A-A_new,'fro')/norm(A, 'fro') eta ]));
%% save the algorithm information and trained projection matrix.
Method.name = 'PCCA';
Method.P=A';
Method.Prob = [];
% Method.Dataname = option.dataname;
Method.Ranking = [];
Method.Dist = [];
Method.Trainoption=option;
return;

% Computing the generalized logistic loss value, the objective function 
% value eq(2). 
function L =logistic_loss(D,y,option)
beta = option.beta;
betaX = beta*(y'.*(D-1));
idx_big = betaX >= 700;
idx_sml = betaX <= -700;
idx_mid = ~idx_big & ~idx_sml;
L = zeros(size(D));
L(idx_mid) = 1/beta*log(1+exp(betaX(idx_mid)));
L(idx_big) = betaX(idx_big)./beta;
L(idx_sml) = 0;
% L =1/beta*log(1+ exp(beta*(y'.*(D-1))));
return;
