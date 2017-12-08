data = 'viper'; %'grid','market','cuhk_detected','prid','ilidsvid'

load(['../FeatureExtraction/feature_' data '_hist_LBP_6patch.mat']);
load(['Split_' data '.mat']);

for s = 1:size(split,1)
    trainID = personID(split(s,1:size(split,2)/2));
    [ix_pos_pair, ix_neg_pair]=GeneratePair(personID,camID,0);
save(['Split_' data '.mat'], 'split',''