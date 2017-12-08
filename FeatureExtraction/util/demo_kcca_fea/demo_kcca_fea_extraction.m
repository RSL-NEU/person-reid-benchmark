%%
% To run this code you have to mex
% + HoG.cpp
% + Include vl_feat library
%   Every images takes less than 0.2 s

%%%%%%%% 1. FEATURE EXTRACTION %%%%%%%%
load('VIPeR_Images.mat');
load('K_no_iso_gaus.mat');
VIPeR_kcca_features = zeros(1264,5138);
for im_id = 1:1264        % the number of samples
    im_id
    im = I{im_id};
    VIPeR_kcca_features(im_id,:) = PETA_cal_img_full_hist(im,K_no_iso_gaus,1);
end

%% %%%%% 2. Building Laplaciang Graph %%%%%%%%%%

p = 316; % number of people
a1 = ones(p,p)*(-1) +1; %*.0001;
a2 = ones(p,p)*(-1) +1; %.0001;
for i = 1:p
    a1(i,i) = 0 ; %1
    a2(i,i) = 1;
end
sW_full= [a1 a2; a2 a1];  imagesc(sW_full), colormap('gray');

DCol = full(sum(sW_full,2));
D = spdiags(DCol,0,speye(size(sW_full,1)));
Lk = D - sW_full;

%%  %%%%%%%%%% 3. Dictionary Learning with Train data %%%%%%%%%%

train_a = VIPeR_kcca_features(1:2:632, :);
train_b = VIPeR_kcca_features(2:2:632, :);
X        = [train_a; train_b];
newfea = (NormalizeFea(double(X)));
[COEFF,pc,~,tsquare] = princomp(newfea,'econ');%         pcadim =  sum(cumsum(latent)/sum(latent)<0.95); %80;%
newfea_pca = pc(:, 1:pca_dim);

%%
%rand('twister',5489);
nBasis = power(2, 7); %316
alpha  = 30;
beta   = .0001;
nIters = 50;
warning('off', 'all');
disp('Learning B ... ');
tic
[B, S, stat] = GraphSC(newfea_pca', Lk, nBasis, alpha, beta, nIters); %'
toc
disp('B is finished.');

% %%%%%%%%%% 4. Testing %%%%%%%%%%
test_a = VIPeR_kcca_features(633:2:end, :)';
test_b = VIPeR_kcca_features(634:2:end, :)';

X_te = [test_a'; test_b'];
newfea_te = (NormalizeFea(double(X_te)));
test_a = newfea_te(1:316,:)';
test_b = newfea_te(317:end,:)';

%%
Lasso = 0;
if Lasso == 1
    % Install SPAMS toolbox for this mode
    param.lambda     = .0001;           % the same as Alpha used in Training
    param.L          = 800;
    param.numThreads = -1;
    param.mode       = 2; 
    param.pos        = 0;
    
    test_b_after_D = full(mexLasso(double(test_a), B,param));
    test_a_after_D = full(mexLasso(double(test_b), B,param));
else % Ridge regression
    lambda1= .1; %.9 for training
    P = pinv( B'*B + lambda1*eye(size(B,2)))*B';
    test_a_after_D = P  * test_a;
    test_b_after_D = P  * test_b;
end

% CMC curve GENERATION
num_p = 316;
maxNumTemplate = 1;
num_gallery    = 316;
num_test       = num_p;

% Permutation indices
idxProbe   = 1:316;     % randperm(num_test);
idxGallery = 1:316;     % randperm(num_gallery);

% Permutation on Train, Gallery and Test set
test_a_after_D = test_a_after_D(:,idxProbe);
test_b_after_D = test_b_after_D(:,idxGallery);

scores_after = pdist2(test_b_after_D', test_a_after_D','cosine');

cmc_nn = zeros(num_gallery,3);
cmc_nn(:,1) = 1:num_gallery;
cmcCurrent = zeros(num_gallery,3);
cmcCurrent(:,1) = 1:num_gallery;

for k=1:num_test
    finalScore = scores_after(:,k);
    [sortScore sortIndex] = sort(finalScore);
    [cmc_nn cmcCurrent] = evaluateCMC_demo(idxProbe(k),idxGallery(sortIndex),cmc_nn,cmcCurrent);
end
recorateY = cmc_nn(: ,2)./cmc_nn(: ,3);
recorateY(1)
%figure(1);hold on;plotCMCcurve(cmc_nn,'g','','VIPeR');



