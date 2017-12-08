% concatanate feature vector in feature_cell
isfirst = 1;
for f = 1:parFea.featurenum
    if parFea.usefeature(f) == 1
        if isfirst == 1
            feature = feature_cell{f,1};
            isfirst = 0;
        else
            feature2 = feature_cell{f,1};  
            feature = [feature feature2];
        end
    end
end


