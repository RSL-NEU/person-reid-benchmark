function [Acc_avg,Acc,Acc_avg_train,Acc_train] = calculate_dist_viper_proj_final(...
    supervec_x,p,lambda1,lambda2,maxit,verbose)

% pre_combine then calculate the distance

load('IDX.mat','-mat');

Acc = zeros(NumTrail,NumTrain);


for j = 1:NumTrail
    
    D = zeros(NumTrain);
    D_train = zeros(NumTrain);
    
    for i = 1:length(supervec_x)
        
        All_x = supervec_x{i};
        
        All_x = normc_safe(All_x);
        
        X_a = All_x(:,idx_cam_a{1});
        X_b = All_x(:,idx_cam_b{1});
        
        X_a_test = X_a(:,idx_test(:,j));
        X_b_test = X_b(:,idx_test(:,j));
        X_a_train = X_a(:,idx_train(:,j));
        X_b_train = X_b(:,idx_train(:,j));
        
        X_train = [X_a_train X_b_train];
        Y_train = [idx_train(:,j);idx_train(:,j)];
        
        %         [M,N,b] = svmml_learn_shiyu2(X_train',Y_train,p,lambda,loss,maxit,verbose);
        %         [A,B,b] = svmml_learn_AB_from_MN2(X_train',Y_train,p,lambda1,lambda2,loss,maxit,verbose,rate);
        %         A = M*M';
        %         B = N*N';
        
        [M,N,b] = svmml_learn_proj_final(X_train',Y_train,p,lambda1,lambda2,maxit,verbose);
        A = M*M';
        B = N*N';
        
        % ensure symmetry for A and B
        A = (A + A')/2;
        B = (B + B')/2;
        
        % now need to calculate the distance function between
        f1 = 0.5*repmat(diag(X_a_test'*A*X_a_test),[1,NumTrain]);
        f2 = 0.5*repmat(diag(X_b_test'*A*X_b_test)',[NumTrain,1]);
        f3 = X_a_test'*B*X_b_test;
        D = D + f1+f2-f3+b;
        
        % calculate the distance for the training data
        f1_train = 0.5*repmat(diag(X_a_train'*A*X_a_train),[1,NumTrain]);
        f2_train = 0.5*repmat(diag(X_b_train'*A*X_b_train)',[NumTrain,1]);
        f3_train = X_a_train'*B*X_b_train;
        D_train = D_train + f1_train+f2_train-f3_train+b;
        
        
    end
    
    %M = zeros(NumTrain);
    R = zeros(NumTrain);
    R_train = zeros(NumTrain);
    
    for i = 1: NumTrain
        [~,IX_D] = sort(D(i,:));
        [~,IX_D_train] = sort(D_train(i,:));
        %M(i,:) = idx_test(IX_D,j)';
        %temp_idx = find(M(i,:) == idx_test(i,j));
        temp_idx = find(IX_D == i);
        temp_idx_train = find(IX_D_train == i);
        R(i,temp_idx:end) = 1;
        R_train(i,temp_idx_train:end) = 1;
    end
    
    Acc(j,:) = mean(R,1); %Acc(j,:) = sum(R,1)./316;
    Acc_train(j,:) = mean(R_train,1); %Acc(j,:) = sum(R,1)./316;
    
    
    Acc_avg = mean(Acc,1);
    Acc_avg_train = mean(Acc_train,1);
    
end