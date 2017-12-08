% Main script to learn projection matrix using a metric learning method

% X - NxF matrix of N feature vectors: each row is a feature vector of
% dimension F
% camID - 1xN array indicating the camera ID of each feature vector in X
% personID - 1xN array indicating the person ID of each feature vector in X
% idx_pos_pair - pre-computed positive pairs
% idx_neg_pair - pre-computed negtive pairs
% mopts - options for metric learning
% dopts - options for dataset

function metric=learnProjectionMatrix(X,camID,personID,idx_pos_pair,idx_neg_pair,mopts,dopts)

X=double(X);
uCamID=unique(camID);
% subsample negtive pairs
if mopts.npratio > 0 % use all negtive pairs when npratio == 0 
    if (strcmp(mopts.method,'pcca') || strcmp(mopts.method,'kpcca') || strcmp(mopts.method,'rpcca') || strcmp(mopts.method,'kmfa'))%...
            %&& strcmp(method.data.name,'market')
        mopts.npratio = 1; % kernel pcca for market will take crazy amount of memoery
    end
    if strcmp(mopts.method,'prdc') % special format for prdc
        idx_pos_pair = [idx_pos_pair; idx_pos_pair(:,2:-1:1)];
        [~,sortidx] = sort(idx_pos_pair(:,1));
        idx_pos_pair = idx_pos_pair(sortidx,:);
%         tmp_neg = zeros(size(idx_pos_pair,1)*method.npratio,2,'uint16');
        tmp_neg = cell(size(idx_pos_pair,1),1);
        parfor i = 1:size(idx_pos_pair,1)
            tmpid = idx_pos_pair(i,1);
            tmpidx = find(idx_neg_pair(:,1)==tmpid | idx_neg_pair(:,2)==tmpid);
            tmp_neg{i} = idx_neg_pair(tmpidx(1:mopts.npratio),:);
%             tmp_neg((i-1)*method.npratio+1:i*method.npratio,:) = ...
%                             idx_neg_pair(tmpidx(1:method.npratio),:);
        end        
        idx_neg_pair = cell2mat(tmp_neg);
    else
        num_neg_pair = min(mopts.npratio*size(idx_pos_pair,1),size(idx_neg_pair,1));
        idx_neg_pair = idx_neg_pair(1:num_neg_pair,:);
    end
end
ix_pair = [idx_pos_pair; idx_neg_pair];
y = [ones(size(idx_pos_pair,1),1); ones(size(idx_neg_pair,1),1).*(-1)];

try
    assert(length(uCamID)==2) % Make sure we are working with data from two cameras
    cam1_id=personID(find(camID==uCamID(1)));
    cam2_id=personID(find(camID==uCamID(2)));
catch
    if strcmp(dopts.name,'airport') % airport dataset has fixed target camera
        camID(camID~=37)=1; 
        uCamID = unique(camID);
    elseif strcmp(mopts.method, 'xqda') % do nothing for xqda
        ;
    else
    %     disp('More than two cameras! May not generate valid results for RankSVM and XQDA!')
        disp('Not exactly two cameras setting, random split the whole dataset into two cameras setting.')
        % better splitting, by Mengran @ 02.28.17
        if numel(uCamID) > 2
            subcid = randsample(numel(uCamID),round(numel(uCamID)/2));
            subID1 = ismember(camID,uCamID(subcid));
            subID2 = ~subID1;
%             camID(subID1) = 1;
%             camID(subID2) = 2;
        else
            randId1 = randsample(size(X,1),round(size(X,1)/2));
            randId2 = setdiff(1:size(X,1),randId1);
            camID(randId1) = 1;
            camID(randId2) = 2;
        end
    end
    uCamID = unique(camID);
    cam1_id=personID(find(camID==uCamID(1)));
    cam2_id=personID(find(camID==uCamID(2)));
end

tsize=length(unique(cam1_id));tsize1=length(unique(cam2_id));
try
    assert(tsize==tsize1);
catch
    disp('Number of people in different cameras is inconsistant');
    tszie = min(tsize,tsize1);
end

% % For KISSME, PCCA, RANKSVM, take the average of the multi-shot data for
% % each person
% if(~strcmp(method.name,'mfa') || ~strcmp(method.name,'lfda'))
%     [X,personID]=avgFeatures(X,personID);
% end

switch lower(mopts.method)
    case 'lfda'
        options.regParam=0.9; % regularization parameter
        options.kNN=7; % NN paramter
        options.d = mopts.d;
        tic;
        T=LFDA(double(X'),personID',options.d,'plain',options.kNN,options.regParam);
        options.time_train = toc;
    case 'fda'
        options.regParam=0.9; % regularization parameter
        options.kNN=7; % NN paramter
        options.d = mopts.d;
        tic;
        T=FDA(double(X'),personID',options.d,'plain',options.kNN,options.regParam);
        options.time_train = toc;
    case 'klfda'
        options.beta =0.01; % regularization parameter
        options.d = mopts.d;
        options.LocalScalingNeighbor =6; % local scaling affinity matrix parameter.
        options.epsilon =1e-4;
        options.kernel = mopts.kernels;
        tic;
        [options]= kLFDA(X, personID',options);
        options.time_train = toc;
        T = options.P;
    case 'pcca'
        if 1
            options.beta =3;  % the parameter in generalized logistic loss function
            options.d = mopts.d; 
            options.epsilon =1e-4; % the tolerence value 
            options.lambda = 0; % set to 0 for pcca
            options.npratio = mopts.npratio;
            tic;
            [options, ~] = oPCCA(X, ix_pair, y, options);
            options.time_train = toc;
            T = options.P;
        else
            delete 'projectionMatrix.out'
            C1 = double(ix_pair-1); % 0 based index 
            C2 = y;
            X = double(X);
    %         [K, Method] = ComputeKernel(X, method.kernel, method);
    %         K = double(K);
            %cd('./PCCA/');
            save('c.out','C1','-ascii');
            save('y.out','C2','-ascii');
            save('x.out','X','-ascii');
            %system('python test.py');
            tic;
            system('python ../MetricLearning/PCCA/pcca.py');
            options.time_train = toc;
            T=textread('projectionMatrix.out');
            options.d = 40;
            options.npratio = mopts.npratio;
            %cd ..
        end
    case 'kpcca'
        if 1
            options.beta =3;  % the parameter in generalized logistic loss function
            options.d = mopts.d; 
            options.epsilon =1e-4; % the tolerence value 
            options.lambda = 0; % set to 0 for pcca
            options.kernel = mopts.kernels;
            options.npratio = mopts.npratio;
            tic;
            [options, ~, ~] = PCCA(X, ix_pair, y, options);
            options.time_train = toc;
            T = options.P;
        else
            delete 'projectionMatrix.out'
            C1 = double(ix_pair-1); % 0 based index 
            C2 = y;
            X = double(X);
            disp(['Begin ', mopts.kernel]);
            [K, options] = ComputeKernel(X, method.kernel, method);
            K = double(K);
            K = K*size(K,1)/trace(K);
            %cd('./PCCA/');
            save('c.out','C1','-ascii');
            save('y.out','C2','-ascii');
            save('x.out','K','-ascii');
            %system('python test.py');
            tic;
            system('python ../MetricLearning/PCCA/pcca_kernel.py');
            options.time_train = toc;
            T=textread('projectionMatrix.out');
            T = T';
            options.d = 40;
            options.npratio = mopts.npratio;
        end
    case 'rpcca'
        if 1
            options.beta =3;  % the parameter in generalized logistic loss function
            options.d = mopts.d; 
            options.epsilon =1e-4; % the tolerence value 
            options.lambda = 0.001;%0.01;
            options.kernel = mopts.kernels;
            options.npratio = mopts.npratio;
            tic;
            [options, ~, ~] = PCCA(X, ix_pair, y, options);
            options.time_train = toc;
            T = options.P;
        else
            delete 'projectionMatrix.out'
            C1 = double(ix_pair-1); % 0 based index 
            C2 = y;
            X = double(X);
            disp(['Begin ', mopts.kernel]);
            [K, options] = ComputeKernel(X, mopts.kernels, method);
            K = double(K);
            K = K*size(K,1)/trace(K);
            %cd('./PCCA/');
            save('c.out','C1','-ascii');
            save('y.out','C2','-ascii');
            save('x.out','K','-ascii');
            %system('python test.py');
            tic;
            system('python ../MetricLearning/PCCA/rpcca.py');
            options.time_train = toc;
            T=textread('projectionMatrix.out');
            T = T';
            options.d = 40;
            options.npratio = mopts.npratio;
        end
    case 'svmml' % locally-adaptive decision functions
        options.p = [];%method.d; %use [] for full matrix
        options.lambda1 = 1e-8;
        options.lambda2 = 1e-6;
        options.maxit = 100;
        options.verbose = 1; % not used
        tic;
        [out] = svmml_learn_full_final(X,personID',options);
        options.time_train = toc;
        T.A = out.A;
        T.B = out.B;
        T.b = out.b;
    case 'prdc' %
        options.Maxloop = 100;
        options.Dimension = 1000;
        options.npratio = mopts.npratio;
        tic
        [out] = LogPenalizedExpRankSubpsace_Seq(X,personID',ix_pair,y,options);
        options.time_train = toc;
        options.is_abs_diff = out.is_abs_diff;
        T = out.P;
    case 'kmfa'
        options.Nw = 0; % 0--use all within class samples
        options.Nb = 0; % number of between class samples 
        options.d = mopts.d; %30;
        options.beta = 0.01;
        options.epsilon =1e-4;
        options.kernel = mopts.kernels;
        tic;
        [options] = MFA(X, personID', options);
        options.time_train = toc;
        T = options.P;
    case 'kissme'
        options.N=size(ix_pair,1); % not used actually
        options.lambda=0.1;
        options.npratio = mopts.npratio;
        cHandle=LearnAlgoKISSME(options);
        tic;
        s=learnPairwise(cHandle,double(X'),ix_pair(:,1),ix_pair(:,2),y>0);
        options.time_train = toc;
        ds.(cHandle.type)=s;
        T=ds.kissme.M;
    case 'itml'
        options.npratio = mopts.npratio;
        cHandle=LearnAlgoITML();
        tic;
        s=learnPairwise(cHandle,X',ix_pair(:,1),ix_pair(:,2),y>0);
        options.time_train = toc;
        ds.(cHandle.type)=s;
        T=ds.itml.M;
    case 'lmnn'
        options.npratio = mopts.npratio;
        
%         cHandle=LearnAlgoLMNN();
        uID = unique(personID);
        for u = 1:numel(uID)
            personID(personID==uID(u)) = u;
        end
        tic;
        [L,Det]=lmnn(X',personID,1,'maxiter',1000,'quiet',1);
        T = L'*L;
%         s=learnPairwise(cHandle,X',ix_pair(:,1),ix_pair(:,2),y>0);
        options.time_train = toc;
%         ds.(cHandle.type)=s;
%         T=ds.lmnn.M;
        %options = [];
    case 'mfa'
        options.intraK=0;
        options.interK=0;
        options.ReducedDim=mopts.d;
        options.Regu = 1; 
        tic;
        [T, ~]=MFA_CDeng(personID, options, X);
        options.time_train = toc;
    case 'ranksvm'
        cam1_features=X(find(camID==uCamID(1)),:);
        cam1_id=personID(find(camID==uCamID(1)));
        cam2_features=X(find(camID==uCamID(2)),:);
        cam2_id=personID(find(camID==uCamID(2)));
        id_common = intersect(unique(cam1_id),unique(cam2_id));
        options.c=0.1; % The C parameter in the RankSVM formulation
        Xt=[];
        count=1;
        for i=1:numel(id_common)
            tmp = cam1_features(cam1_id==id_common(i),:)';
            Xt(:,count)=mean(tmp,2);
            tmp = cam2_features(cam2_id==id_common(i),:)';
            Xt(:,count+1)=mean(tmp,2);
            count=count+2;
        end
        parm.TrainFeatures=Xt;
        %parm.TrainImagesNumPerPerson=2*ones(1,tsize);
        parm.TrainImagesNumPerPerson=2*ones(1,size(Xt,2)/2);
        args=strcat({' '},'-c',{' '},num2str(options.c)); % Linear SVM formulation
        tic;
        T=svm_struct_learn(args{1},parm);
        options.time_train = toc;
    case 'xqda'
%         cam1_features=X(camID==uCamID(1),:);
%         cam1_id=personID(camID==uCamID(1));
%         cam2_features=X(camID==uCamID(2),:);
%         cam2_id=personID(camID==uCamID(2));
        
        [cam1_features, cam2_features, cam1_id, cam2_id] = gen_train_sample_xqda(personID', camID', X');
        options = [];
        tic;
        [T.W, T.M] = XQDA(cam1_features, cam2_features, cam1_id, cam2_id);
        options.time_train = toc;
    case 'mlapg'
        cam1_features=X(find(camID==uCamID(1)),:);
        cam1_id=personID(find(camID==uCamID(1)));
        cam2_features=X(find(camID==uCamID(2)),:);
        cam2_id=personID(find(camID==uCamID(2)));
        options = [];
        tic;
        [T] = MLAPG(cam1_features, cam2_features, cam1_id', cam2_id');
        options.time_train = toc;
    case 'nfst'
        tic;
        if 0%gpuDeviceCount > 0 && any(size(X)>1e4) % use GPU for really large matrix
            try 
                reset(gpuDevice()); % reset GPU memory if used
                X = gpuArray(X);
                [K, options] = ComputeKernel(X, method.kernel, method);
            catch
                X = gather(X);
                disp('Compute on GPU failed, using CPU now...');
                [K, options] = ComputeKernel(X, method.kernel, method);
            end
            reset(gpuDevice()); % reset GPU memory if used
        else 
            [K, options] = ComputeKernel(X, mopts.kernels, mopts);
        end
        T = NFST(K, personID');
        T = T';
        options.time_train = toc;
        options.kernel = mopts.kernels;
    case 'kcca'
        cam1_features=X(camID==uCamID(1),:);
        cam1_id=personID(camID==uCamID(1));
        cam2_features=X(camID==uCamID(2),:);
        cam2_id=personID(camID==uCamID(2));
        
        if length(cam1_id)~=length(cam2_id) % has to be the same dimension
            if length(cam1_id) > length(cam2_id)
                tmp_num = length(cam2_id);
                tmp_subidx = randsample(length(cam1_id),tmp_num);
                cam1_features = cam1_features(tmp_subidx,:);
                cam1_id = cam1_id(tmp_subidx);
            else
                tmp_num = length(cam1_id);
                tmp_subidx = randsample(length(cam2_id),tmp_num);
                cam2_features = cam2_features(tmp_subidx,:);
                cam2_id = cam2_id(tmp_subidx);
            end
        end
        
        eta_kcca = 1;
        kapa_kcca = 0.5;
        tic;
        [cam1_ker, ~] = ComputeKernel(cam1_features, mopts.kernels, mopts);
        [cam2_ker, options] = ComputeKernel(cam2_features, mopts.kernels, mopts);
        [Wx, Wy, ~] = kcanonca_reg_ver2(cam2_ker,cam1_ker,eta_kcca,kapa_kcca,0,0);
        T.Wx = Wx;
        T.Wy = Wy;
        T.cam1_ker = cam1_ker;
        T.cam2_ker = cam2_ker;
        T.cam1_feat = cam1_features;
        T.cam2_feat = cam2_features;
        options.time_train = toc;
        options.kernel = mopts.kernels;
    case 'sssvm'
        par = struct(...
                    'train_svm_c',   300, ...   % sample-specific svm  C  for positive set
                    'wpos',          0.1, ...   % sample-specific wpos*C  for negative set
                    'nIter',         100, ...   % Iteration Number for LSSCDL
                    'lambda1',       0.1,...    % lambda1*||Alphaw -Mx * Alphax||^2
                    'lambdac',       0.01,...   % lambdac*||Alphaw||^2, lambdac*||Alphax||^2 
                    'lambdam',       0.01,...   % lambdam*||Mx||^2
                    'lambdad',       0.01,...   % lambdad*||Dx||^2, lambdad*||Dw||^2
                    'epsilon',       5e-3);     % convergence 
        cam1_features=X(camID==uCamID(1),:)';
        cam1_id=personID(camID==uCamID(1))';
        cam2_features=X(camID==uCamID(2),:)';
        cam2_id=personID(camID==uCamID(2))';
        tic;
        T.P = LDA(cam2_features', cam1_features', cam1_id, cam2_id);
        cam1_features = T.P*cam1_features;
        cam2_features = T.P*cam2_features;
        par.K = numel(cam1_id);
        [W, ~]=SpecificSVMLearn(cam1_features, cam2_features, cam1_id, cam2_id,par);
        T.Dict=JointDictLearning(cam1_features, W,par); 
        T.par = par;
        options.time_train = toc;
end
metric.name=mopts.name;
metric.T=T;
metric.options = options;





