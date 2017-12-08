% VipeR Main 
% This is the main function for running VipeR 

% Copyright (C) 2013 by Shiyu Chang and Zhen Li.

clear; close all; 
dbstop if error;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% add path to the current directory
cur_dir = pwd;

% add feature path
data_path{1} = [cur_dir '/VIPeR_Feature/feature1'];
data_path{2} = [cur_dir '/VIPeR_Feature/feature2'];
data_path{3} = [cur_dir '/VIPeR_Feature/feature3'];

num_combine = length(data_path);

% add code path
addpath([cur_dir '/code']);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Control Parameters
% PARTITION_DATA: 1:partite the data 
%                 0:load it
% GENERATE_TRAIN: 1:generate
%                 0:load it 
% D0_PCA: 1:train PCA projection,
%         0:load it
%        -1: skip  
% CAL_DIST: 1: learn full matrix M
%           2: learn projection matrix L
%

PARTITION_DATA = 1;
GENERATE_Train = 1; 
DO_PCA = 0;
CAL_DIST = 1; % default set as 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Preset pipeline parameters 
% partition data
% Do not change these file name 
if (PARTITION_DATA == 1)
    FileName = cell(num_combine,1);
    FileName{1} = 'realfilelist_Img_hmm_mix128_reduce2_dim40';
    FileName{2} = 'realfilelist_Img_hmm_mix128_reduce0_dim36';
    FileName{3} = 'realfilelist_Img_hmm_mix128_reduce2_dim40';
end

% generating training data
if (GENERATE_Train == 1)
    NumTrain = 316;
    NumTrail = 1;
end

% Do PCA
if (DO_PCA == 1)
    Pyramid_idx = [0,1,2,3,4,5,6,7];
    data_name{1} = 'Img_hmm_mix128_reduce2_dim40.mat.';
    data_name{2} = 'Img_hmm_mix128_reduce0_dim36.mat.';
    data_name{3} = 'Img_hmm_mix128_reduce2_dim40.mat.';
    % What is the PCA_dimension you want to reduce to
    PCA_dim = 1000;
end


% Calculating Pairwise Distance
if (CAL_DIST)
    Pyramid_idx = [0,1,2,3,4,5,6,7];
    data_name{1} = 'Img_hmm_mix128_reduce2_dim40.mat.';
    data_name{2} = 'Img_hmm_mix128_reduce0_dim36.mat.';
    data_name{3} = 'Img_hmm_mix128_reduce2_dim40.mat.';
    weight = [1,1,1,1,1,1,1,1];
    p = 600;
    lambda1 = 0;
    lambda2 = 0;
    loss = 2;
    maxit = 300;
    verbose = 1;
    rate = 1;
end



%% partition data and alginment
% This part of partition of data, only spilt the cam_a and cam_b; 
% after is part is done.  the idx of cam_a and cam_b from the HG
% supervector will be obtained

fprintf('==> Partition Data: \n');

if (PARTITION_DATA == 1)
    
    for i = 1:num_combine
        
        fileread_info = textread([data_path{i} '/' FileName{i}], '%s', 'delimiter', '\n');
        [idx_cam_a{i},idx_cam_b{i}] = partition(fileread_info);
    end
    
    % check among cam_a and cam_b for the idx of different features
    
    if (num_combine ~= 1)
        
        for i = 1:num_combine-1
            if (~isequal(idx_cam_a{i},idx_cam_a{i+1}) || ~isequal(idx_cam_b{i},idx_cam_b{i+1}))
                error ('Two different feature idx are not allgined\n');
            end
        end
    end
    
    save('IDX.mat','-mat','idx_cam_a','idx_cam_b');
    
elseif (PARTITION_DATA == 0)
    load('IDX.mat','-mat','idx_cam_a','idx_cam_b');
elseif (PARTITION_DATA == -1)
    fprintf('====> Partition Data Skipped \n');
else
    error('====> No such options \n');
end
fprintf('==> Partition Data Done \n');


%% Generating Training and Testing Data Idex

fprintf('==> Generating Training and Testing Data: \n');
if (GENERATE_Train == 1)
    idx_train = zeros(NumTrain,NumTrail);
    idx_test = zeros(NumTrain,NumTrail);
    rand('seed', 20120501);
    for i = 1:NumTrail
        length_cam = length(idx_cam_a{1});
        perm_cam = randperm(length_cam);
        idx_train(:,i) = perm_cam(1:NumTrain);
        idx_test(:,i) =  perm_cam(NumTrain+1:end);
    end
    save('IDX.mat','-append','idx_train','idx_test','NumTrain','NumTrail');
elseif (GENERATE_Train == 0)
    load('IDX.mat','-mat','idx_train','idx_test','NumTrain','NumTrail');
elseif (GENERATE_Train == -1)
    fprintf('====> Generating Training and Testing Data Skipped \n');
else
    error('====> No such options \n');    
end
fprintf('==> Generating Training and Testing Data Done \n');    


%% Do PCA
% Learn PCA projection matrix on training set, and then apply to all data.
% (Training and Testing are seperated)

fprintf('==> Calculating PCA Projection: \n');
    
if(DO_PCA == 1) 
    
    supervec_x = cell(1,num_combine);
    featureWOPCA = [];
    for i = 1:num_combine
        temp_x = [];
        for j = 1:length(Pyramid_idx)
            % combine all the pyramid together
            load ([data_path{i} '/' data_name{i} num2str(Pyramid_idx(j))],'-mat','AllImgs');
            temp_x = [temp_x;AllImgs];
        end 
%         supervec_x{i} = do_PCA(temp_x,PCA_dim,1);
        featureWOPCA = [featureWOPCA;temp_x];
    end
    save('supervec.mat','-mat','supervec_x');
       
elseif(DO_PCA == 0)
    load('supervec.mat','-mat','supervec_x');
end

fprintf('==> Calculating PCA Projection Done: \n');

%% Calculating Pairwise Distance

% Calculating Pairwise Distance
fprintf('==> Calculating Pairwise Distance: \n');

if (CAL_DIST == 1)

[Acc_avg,Acc,Acc_avg_train,Acc_train] = calculate_dist_viper_full_final(supervec_x,...
    p,lambda1,lambda2,maxit,verbose);
elseif (CAL_DIST == 2)
    
[Acc_avg,Acc,Acc_avg_train,Acc_train] = calculate_dist_viper_proj_final(supervec_x,...
    p,lambda1,lambda2,maxit,verbose);    
else
    error('Invalid options');
end

%% Display the final result 

[Acc_avg_train(1) Acc_avg_train(10) Acc_avg_train(20) Acc_avg_train(50)]

[Acc_avg(1) Acc_avg(10) Acc_avg(20) Acc_avg(50)]

[Acc(1,1) Acc(1,10) Acc(1,20) Acc(1,50)]
% 



% 
% if (CAL_DIST == 1)  
%     [Acc_avg,Acc] = calculate_dist_multi_PCA_matric(supervec_x,p,lambda,loss,maxit,verbose);
% %     [Acc_avg,Acc] = calculate_dist_multi_PCA(supervec_x);
% elseif (CAL_DIST ==2)
%     [Acc_avg,Acc] = calculate_dist_multi_PCA_lmnn(supervec_x,p);
% elseif (CAL_DIST == 3) 
%     [Acc_avg,Acc] = calculate_dist_multi_PCA_ITML(supervec_x,p,lambda,loss,maxit,verbose);
% elseif (CAL_DIST == 4)  
%     [Acc_avg,Acc,Acc_avg_train,Acc_train] = calculate_dist_viper_new(supervec_x,p,lambda1,lambda2,loss,maxit,verbose,rate);
% end
% fprintf('==> Calculating Pairwise Distance Done: \n');
% 

