%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% extract_features.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
config;
fprintf('*** low level feature extraction *** \n');
fprintf('database = %s \n', databasename );

% extract feature for all images.
fprintf('+ extract feature for all images \n');
for f = 1:parFea.featurenum
    if parFea.usefeature(f) == 1
        param = parFea.featureConf{f};
        fprintf('feature = %d [ %s ] \n', f, param.name);
        feature_all = zeros( allimagenums, param.dimension );
        
        t0 = tic;
        for imgind = 1:allimagenums
            if mod(imgind, 100) == 0; fprintf('imgind = %d / %d \n', imgind, allimagenums); end
            X = imread( strcat(datadirname, allimagenames{imgind})); % load image
            if size(X, 1) ~= H0 || size(X, 2) ~= W0; X = imresize(X, [H0 W0]); end % resize
            
            
            feature_all(imgind, :) = GOG(X, param); % extract GOG
            
        end
        feaTime = toc(t0);
        meanTime = feaTime/allimagenums;
        fprintf('mean feature extraction time %.3f seconds per image\n', meanTime);
        
        name1 = sprintf('feature_all_%s', param.name );
        name = strcat( featuredirname, databasename, '_',  name1, '.mat');
        fprintf('%s \n', name);
        
        save( name,  'feature_all', '-v7.3' );
    end
end






