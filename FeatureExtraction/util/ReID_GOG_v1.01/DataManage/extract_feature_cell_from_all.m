% extract feature cells for particular training/test division
feature_cell = cell( parFea.featurenum, 1);
if tot == 1
    numimages_train = numel(traininds_set{set});
    for f = 1:parFea.featurenum
        if parFea.usefeature(f) == 1
            feature_cell(f,1) = {zeros( numimages_train, size(feature_cell_all{f,1}, 2)) };
            for ind = 1:numimages_train
                feature_cell{f,1}(ind,:) = feature_cell_all{f,1}( traininds_set{set}(ind), :);
            end
        end
    end
    clear numimages_train;
end
if tot == 2
    numimages_test = numel(testinds_set{set});
    for f = 1:parFea.featurenum
        if parFea.usefeature(f) == 1
            feature_cell(f,1) = {zeros( numimages_test,  size(feature_cell_all{f,1}, 2)) };
            for ind = 1:numimages_test
                feature_cell{f,1}(ind,:) = feature_cell_all{f,1}( testinds_set{set}(ind), :);
            end
        end
    end
    clear numimages_test;
end

