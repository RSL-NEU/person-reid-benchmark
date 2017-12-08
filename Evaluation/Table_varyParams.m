clc
clear 
%% 
pca = 0; % vary pca dimensions
np = 1; % vary num of patches
if pca 
    %% single 
    rank1_all = [];
    datasets = {'viper'};
    pcad = [50 100 150 200 250 300];
    for d = 1:numel(datasets)
        dataname = datasets{d};
        for p = 1:numel(pcad)
            pcadim = pcad(p);
            script_result_load;
            rank1_all(:,:,p) = rank1;
        end
    end
    %% Plot
    figure
    rank1_all  = permute(rank1_all,[3,1,2]);
    rank1_all = reshape(rank1_all,[numel(pcad)*numel(featname),numel(algoname)]);
    figure,imagesc(rank1_all);
    colormap jet
    hold on

    ax = gca;
    ax.XTickLabel = algoname;
    ax.XTick = 1:numel(algoname);
    ax.XTickLabelRotation = 45;

    ax.YTick = round(numel(pcad)/2):numel(pcad):size(rank1_all,1);
    ax.YTickLabel = featname;
    colorbar
    for l = 1:numel(featname)-1
        line([0.5,size(rank1_all,2)+0.5],[l*numel(pcad)+0.5,l*numel(pcad)+0.5],...
            'LineWidth',4,'Color','w');
    end
elseif np
    res_folder = '../Results/numP';
    %% single 
    datasets = {'airport','viper','market','hda','cuhk_detected','grid','3dpes'};
    pcadim = 100;
    algoname = {'klfda_exp'};
    featname = {'lomo'};
    np = [6 9 15 24];
    rank1_all = nan(numel(np),numel(dataset));
    for d = 1:numel(datasets)
        dataname = datasets{d};
        for p = 1:numel(np)
            numP = np(p);
            script_result_load;
            rank1_all(p,d) = rank1;
        end
    end
end
    
