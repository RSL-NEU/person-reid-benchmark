clc
clear 
%%
dataset = 'DukeReIDpair';
resfolder = '../Results';

feat = 'gog';
algo = 'klfda_exp';

% load result
info = dir(sprintf('%s/rank_%s_%s_%s_*2017*.mat',resfolder,dataset,algo,feat));
if numel(info) > 1
    disp('Non unique results!');
end
load(fullfile(resfolder,info(1).name));

rank1 = zeros(8,8);
mAPs = zeros(8,8);

for i = 1:8
    for j = 1:8
        tmpidx = all(bsxfun(@eq,metric.cam_pair,[i,j]),2);
        if ~any(tmpidx)
            continue;
        end
        rank1(i,j) = metric.rank(tmpidx,1);
        mAPs(i,j) = metric.mAP(tmpidx);
    end
end

rank1 = rank1.*100;