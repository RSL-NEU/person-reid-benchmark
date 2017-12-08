function dis = singleShotRanking(probeFeat, galleryFeat ,metric,mopts)
% single shot distance computation
% ----INPUT----
% probeFeat     - [NxD] features for N probe samples
% galleryFeat   - [MxD] features for M gallery samples
% metric        - [struct] learned metric
% mopts         - [struct] metric learning options
% ----OUTPUT----
% dis           - [MxN] pair-wise distance matrix
% Write by Mengran Gou @ 2017



if strcmp(mopts.method,'svmml')
    A = metric.T.A;
    B = metric.T.B;
    b = metric.T.b;
    max_dim = max(sum(idx_probe(pr,:)),sum(idx_gallery(pr,:)));
    f1 = zeros(max_dim);
    f2 = zeros(max_dim);
    f3 = zeros(max_dim);
    f1 = 0.5*repmat(diag(probeFeat*A*probeFeat'),[1,sum(idx_gallery(pr,:))]);
    f2 = 0.5*repmat(diag(galleryFeat*A*galleryFeat')',[sum(idx_probe(pr,:)),1]);
    f3 = probeFeat*B*galleryFeat';
    dis = f1+f2-f3+b;
elseif strcmp(mopts.method,'kissme') || ...
        strcmp(mopts.method,'itml') || ...
        strcmp(mopts.method,'lmnn')
    M = metric.T;
    dis = sqdist(probeFeat',galleryFeat',M);
elseif(strcmp(mopts.method,'ranksvm'))
    for pr1=1:size(probeFeat,1)
        for pr2=1:size(galleryFeat,1)
            probeFeat1=probeFeat(pr1,:);
            galleryFeat1=galleryFeat(pr2,:);
            dis(pr1,pr2)=abs(bsxfun(@minus,probeFeat1,galleryFeat1))*metric.T;
        end
    end
elseif strcmp(mopts.method,'xqda')
    W = metric.T.W;
    M = metric.T.M;
    dis = MahDist(M, probeFeat * W, galleryFeat * W);
elseif strcmp(mopts.method,'prdc')
    P = metric.T;
    for p = 1:size(probeFeat,1)
        [DiffSetC] = MakeDiffSubset(galleryFeat',probeFeat(p,:)',metric.options.is_abs_diff,1);
        dis(p,:) = sum((P'*DiffSetC).^2,1);
    end
elseif strcmp(mopts.method,'kcca')
    K.kernel = metric.options.kernel;
    K.rbf_sigma = metric.options.rbf_sigma;
    % compute kernel seperately
    prob_k = ComputeKernelTest(metric.T.cam1_feat, probeFeat, K);
    gal_k = ComputeKernelTest(metric.T.cam2_feat, galleryFeat, K);
    prob_k = prob_k';
    gal_k = gal_k';
    % centering kernel
    [~,test_a_ker,~,test_b_ker] = ...
        center_kcca(metric.T.cam1_ker,prob_k,metric.T.cam2_ker,gal_k);
    test_b_ker_proj = test_b_ker*metric.T.Wx;
    test_a_ker_proj = test_a_ker*metric.T.Wy;
    dis = pdist2(test_a_ker_proj,test_b_ker_proj,'cosine');
elseif strcmp(mopts.method,'sssvm')
    dis = MatchScore(probeFeat',galleryFeat',metric.T.Dict,metric.T.par);
    dis = dis';
else
    dis = pdist2(probeFeat,galleryFeat,'euclidean');
end