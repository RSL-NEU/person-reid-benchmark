clc
clear 
%% single
best = [];
bname = {};

% fix_feat = 1;
% fix_algo = 0;
% plot_PUR = [];
% plot_r1 = [];
%datasets = {'viper','grid','3dpes','cuhk_detected','hda','market','airport'};
% datasets={'airport'};
% for d = 1:numel(datasets)
%     dataname = datasets{d};
%     script_result_load;
%     [~,tmp] = max(CMC(:,1));
%     best = [best; CMC(tmp,1:20)];
%     bname = [bname, names{tmp}];
% end

%% multi
% datasets={'prid','v47','caviar','ward_12','ward_13','saivt_38','saivt_58',...
%     'raid_12','raid_13','raid_14','ilidsvid'};
datasets={'prid'};
for d = 1:numel(datasets)
    dataname = datasets{d};
    script_result_load_multi;
    [~,tmp] = max(CMC(:,1));
    best = [best; CMC(tmp,1:20)];
    bname = [bname, names{tmp}];
end