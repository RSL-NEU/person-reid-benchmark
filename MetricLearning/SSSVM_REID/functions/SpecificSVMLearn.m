function [W,Response] = SpecificSVMLearn(X1, X2,X1_label,X2_label,par)
%% function: Learn the distance weight for each item of X1
%% input:
%   X1 : D x N matrix
%   X2 : D x M matrix
%   X1_label: person identity for X1
%   X2_label: person identity for X2
%  par : parameters for libsvm
%% output:
%   W : D x N matrix
%   Response: svm response value

%% 
fprintf('Training Sample-Specific SVMs...');  
[dim N1]=size(X1);
[dim N2]=size(X2);
W=zeros(3*dim,N1);
Response=zeros(N1,N2);

for i=1:N1
    ind=X2_label==X1_label(i);
    label=-1*ones(N2,1);
    label(ind)=1;
    Diff=[repmat(X1(:,i),1,N2);abs(repmat(X1(:,i),1,N2)-X2);X2]; 
%     svm_model = svmtrain(label, Diff',sprintf(['-s 0 -t 0 -c' ...
%                     ' %f -w1 %.9f -q'], par.train_svm_c, par.wpos)); 
    svm_model = libsvmtrain(label, Diff',sprintf(['-s 0 -t 0 -c' ...
                    ' %f -w1 %.9f -q'], par.train_svm_c, par.wpos)); 
%     [predict_label, accuracy, dec_values]= libsvmpredict(label, Diff', svm_model);            
    %convert support vectors to decision boundary
    svm_weights = full(sum(svm_model.SVs .* ...
                         repmat(svm_model.sv_coef,1, ...
                                size(svm_model.SVs,2)),1));
   
   Resp=svm_weights*Diff; 
   % make sure that the response value for matched pair is smaller than
   % unmatched ones
   ind1=find(ind==1); inds1=ind1((randsrc(1,1,randperm(length(ind1)))));
   ind0=find(ind==0); inds0=ind0((randsrc(1,1,randperm(length(ind0)))));
   if Resp(inds1)>Resp(inds0) 
%        i
       svm_weights=-svm_weights;
   end
   W(:,i)=svm_weights';  
   Resp=svm_weights*Diff; 
   Response(i,:)=Resp';
end
fprintf('Done!\n');
end 