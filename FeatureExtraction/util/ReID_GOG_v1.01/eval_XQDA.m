%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% eval_XQDA.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
config;
load_features_all; % load all features.

CMCs = zeros( sys.setnum, numperson_garalley );
trainnum = numperson_train;

for set = 1:sys.setnum
    fprintf('----------------------------------------------------------------------------------------------------\n');
    fprintf('set = %d \n', set);
    fprintf('----------------------------------------------------------------------------------------------------\n');
    
    %% Training data
    tot = 1;
    extract_feature_cell_from_all;  % load training data
    apply_normalization; % feature normalization
    conc_feature_cell; % feature concatenation
    
    % train XQDA metric learning
    camIDs = traincamIDs_set{set};
    probX = feature(camIDs == 1, :);
    galX = feature(camIDs == 2, :);
    labels = trainlabels_set{set};
    probLabels = labels(camIDs == 1);
    galLabels = labels(camIDs == 2);
    
    [XQDAresult.W, XQDAresult.M, inCov, exCov] = XQDA(galX, probX, galLabels, probLabels);
    clear camIDs probX galX probX galLabels probLabels
    
    %% Test data
    tot = 2;
    extract_feature_cell_from_all; % load test data
    apply_normalization; % feature normalization
    conc_feature_cell; % feature concatenation
    
    % apply XQDA metric learning
    camIDs = testcamIDs_set{set};
    probX = feature(camIDs == 1, :);
    galX = feature(camIDs == 2, :);
    labels = testlabels_set{set};
    labelsPr = labels(camIDs == 1);
    labelsGa = labels(camIDs == 2);
    
    if sys.database ~= 3
        % single shot matching
        scores = MahDist(XQDAresult.M, galX * XQDAresult.W, probX * XQDAresult.W)';
        
        CMC = zeros( numel(labelsGa), 1);
        for p=1:numel(labelsPr)
            score = scores(p, :);
            [sortscore, ind] = sort(score, 'ascend');
            
            correctind = find( labelsGa(ind) == labelsPr(p));
            CMC(correctind:end) = CMC(correctind:end) + 1;
        end
        CMC = 100.*CMC/numel(labelsPr);
        CMCs(set, :) = CMC;
        
    else
        % multishot matching for CUHK01
        probX1 = probX(1:2:size(probX, 1), :);
        probX2 = probX(2:2:size(probX, 1), :);
        galX1 = galX(1:2:size(galX, 1), :);
        galX2 = galX(2:2:size(galX, 1), :);
        
        labelsPr1 = labelsPr(1:2:size(probX, 1), 1);
        labelsPr2 = labelsPr(2:2:size(probX, 1), 1);
        labelsGa1 = labelsGa(1:2:size(galX, 1), 1);
        labelsGa2 = labelsGa(2:2:size(galX, 1), 1);
        scores1 = MahDist(XQDAresult.M, galX1 * XQDAresult.W, probX1 * XQDAresult.W)';
        scores2 = MahDist(XQDAresult.M, galX2 * XQDAresult.W, probX1 * XQDAresult.W)';
        scores3 = MahDist(XQDAresult.M, galX1 * XQDAresult.W, probX2 * XQDAresult.W)';
        scores4 = MahDist(XQDAresult.M, galX2 * XQDAresult.W, probX2 * XQDAresult.W)';
        
        scores = scores1 + scores2 + scores3 + scores4;
        
        CMC = zeros( numel(labelsGa1), 1);
        for p=1:numel(labelsPr1)
            score = scores(p, :);
            [sortscore, ind] = sort(score, 'ascend');
            
            correctind = find( labelsGa1(ind) == labelsPr1(p));
            CMC(correctind:end) = CMC(correctind:end) + 1;
        end
        CMC = 100.*CMC/numel(labelsPr1);
        CMCs(set, :) = CMC;
    end
    
    fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
    fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', CMC([1,5,10,15,20]));
    clear camIDs probX galX probX galLabels probLabels options XQDAresult
    
end

fprintf('----------------------------------------------------------------------------------------------------\n');
fprintf('  Mean Result \n');
fprintf('----------------------------------------------------------------------------------------------------\n');
clear set;

figure('Position',[200 200 400 300]);
set(gcf, 'color', 'white');
set(gcf,'defaultAxesFontSize',10);
set(gcf,'defaultAxesFontName','Arial');
set(gcf,'defaultAxesFontWeight','demi');
set(gcf,'defaultTextFontSize',10);
set(gcf,'defaultTextFontName','Arial');
set(gcf,'defaultTextFontWeight','demi');

CMC = mean( squeeze(CMCs(1:sys.setnum , :)), 1);
fprintf(' Rank1, Rank5, Rank10, Rank20 \n');
fprintf('%5.1f%%, %5.1f%%, %5.1f%%, %5.1f%% \n', CMC([1,5,10,20]) );

semilogx( CMC, '-', 'color', 'r', 'LineWidth', 5 );
axis([1  min(100, numperson_garalley) 0 100]);
hold on;
grid on;

title(databasename, 'fontsize', 12);
xlabel('Rank score', 'fontsize', 12);
ylabel('Recognition percentage', 'fontsize', 12);
grid on;




