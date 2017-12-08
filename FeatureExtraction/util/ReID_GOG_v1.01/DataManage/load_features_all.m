disp('*** load all extracted features ***');
feature_cell_all = cell( parFea.featurenum, 1);
for f = 1:parFea.featurenum
    if parFea.usefeature(f) == 1
        fprintf('feature = %d [ %s ] \n', f, parFea.featureConf{f}.name);
    
        name1 = sprintf('feature_all_%s', parFea.featureConf{f}.name);
        name = strcat( featuredirname, databasename, '_',  name1, '.mat');
       
        fprintf('%s \n', name);
        load( name, 'feature_all');
        feature_cell_all{f,1} = feature_all;
        clear feature_all;
    end
end


