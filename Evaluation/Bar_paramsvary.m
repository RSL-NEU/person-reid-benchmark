clc
clear 
close all

featname = 'LOMO';
dataname= {'viper','market','airport','cuhk_detected','grid','hda','3dpes'};
    
% dataname = {
%     'saivt_38','saivt_58','ilidsvid','prid',...
%     'ward_12','ward_13','raid_12','raid_13',...
%     'raid_14','caviar','v47'};

metricNameSingle='klfda_exp';
metricNameMulti='kissme';
rankName='srid';

% With PCA results

np=[6 9 15 24];

for i=1:numel(dataname)
    % Each row corresponds to a particular dataset
    for j=1:numel(np)
        if(strcmp(dataname{i},'viper')||strcmp(dataname{i},'airport')||strcmp(dataname{i},'market')...
                ||strcmp(dataname{i},'cuhk_detected')||strcmp(dataname{i},'grid')||strcmp(dataname{i},'hda')...
                ||strcmp(dataname{i},'3dpes'))
            info = dir(strcat('../Results/numP/rank_',dataname{i},'_',metricNameSingle,'_',lower(featname),...
                '_p100_s',num2str(np(j)),'*.mat'));
        else
            info = dir(strcat('../Results/numP/rank_',dataname{i},'_clustering_',metricNameMulti,'_',rankName,...
                '_',featname,'_p100_s',num2str(np(j)),'*.mat'));
        end
        rank=load(fullfile(info.name));rank=rank.rank;
        output(i,j)=rank(1);
    end    
end

% Without PCA results

for i=1:numel(dataname)
    if(strcmp(dataname{i},'viper')||strcmp(dataname{i},'airport')||strcmp(dataname{i},'market')...
            ||strcmp(dataname{i},'cuhk_detected')||strcmp(dataname{i},'grid')||strcmp(dataname{i},'hda')...
            ||strcmp(dataname{i},'3dpes'))
        info = dir(strcat('../Results/rank_',dataname{i},'_',metricNameSingle,'_',lower(featname),'_2016_','*.mat'));
    else
        info = dir(strcat('../Results/rank_',dataname{i},'_clustering_',metricNameMulti,'_',rankName,...
            '_',featname,'_2016_','*.mat'));
    end
    if(numel(info)>1) info=info(1);end
    rank=load(fullfile(info.name));rank=rank.rank;
    output1(i,1)=rank(1);
end

output=[output1 output];
%%
color_marker;
figure;
b = bar(output);
for i = 1:numel(b)
    b(i).FaceColor = p_color(i,:);
end
xlabelp = datasetMap(dataname);
xlabelp = cellfun(@(X) ['\bf{' X '}'],xlabelp,'uni',0);
xlim([0.5,numel(xlabelp)+0.5]);
ylim([0 100]);
ax = gca;
ax.XTick = 1:numel(xlabelp);
ax.XTickLabel = xlabelp;
ax.XTickLabelRotation = 45;
ax.TickLabelInterpreter = 'latex';
set(gca,'FontSize',20);
legname = mat2cell(np',ones(numel(np),1),1);
legname = cellfun(@(X) ['\bf{patch ' num2str(X) '; PCA}'],legname,'uni',0);
legname = ['\bf{patch 6; NO PCA}'; legname];
legend(legname,'FontSize',20,'interpreter','latex','location','best');
ylabel('\bf{Matching Rate (\%)}','FontSize', 20, 'FontWeight', 'bold');
grid on
box on
if strcmp(lower(dataname{1}),'viper')
    title(['\bf{PCA and number of patches: single-shot}'],'FontSize',20,'FontWeight','bold','interpreter','latex');
    namep = 'Bar_pca_patches_single';
else
    title(['\bf{PCA and number of patches: multi-shot}'],'FontSize',20,'FontWeight','bold','interpreter','latex');
    namep = 'Bar_pca_patches_multi';
end

fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 9 8];
fig.PaperPositionMode = 'manual';

%print(namep,'-depsc','-tiff','-r300')

% save('PCA_numPatch_rank1_results.dat','output','-ascii');