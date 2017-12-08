clear;
addpath('../../common/');
load('../../../Matlab_Data/FeatureMat/CLE_features_selected.mat');

max_gallery_shot_num = 1; % number of gallery shot for each individual
max_prob_shot_num = 999; % number of probe shot for each individual
max_train_shot_num = 10;
train_size = 45;
person_num = numel(ImagesNumPerPerson);
train_ind_upper_bound = person_num;
real_world_experiment = false;
ImagesNumPerPerson = ImagesNumPerPerson';
preProcessDatasetFeatures;

parm.TrainFeatures = FeatureSet(:, abs_train_ind);
parm.TrainImagesNumPerPerson = TrainImagesNumPerPerson;
svm_model = svm_struct_learn(' -c 0.001', parm);

save('svm_model_CLE1.mat', 'svm_model');
%% test 
gallery_feature = FeatureSet(:,abs_gallery_ind);
prob_img_per_person = ImagesNumPerPerson(prob_ind);
total_test_num = 0;
rank = zeros(gallery_size, 1);
for i = 1 : prob_size
    person_feature = getPersonFeature(FeatureSet, prob_ind(i), ImagesNumPerPerson);
    person_feature(:, gallery_shot_ind(i)) = [];
    for j = 1 : prob_img_per_person(i) - 1
        F = person_feature(:,j);
        score = svm_model' * abs(bsxfun(@minus, gallery_feature, F));
        [~, sortedInd] = sort(score, 'descend');
        pos = find(sortedInd == i);
        rank(pos) = rank(pos) + 1;
        total_test_num = total_test_num + 1;
    end
end
rank = cumsum(rank)/total_test_num*100;

%% print out result
fprintf('Rank    SVM    Discriminative\n');
for i = 1 : 20
    fprintf('%4d    %4.2f%%    \n', i, rank(i));
end