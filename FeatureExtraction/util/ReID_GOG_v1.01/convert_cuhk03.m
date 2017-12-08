%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% convet_cuhk03.m
%% covert cuhk03.m into image files 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
datadirname = 'E:/database_reid/cuhk03_release/cuhk03_release/'; % Path for cuhk_03.mat
savedatadirname = 'E:/database_reid/CUHK03/'; % Path for save folder

disp('load chuk-03.mat');
load( strcat(datadirname, 'cuhk-03.mat'));
disp('end load');

% labeled
disp('labeled');
for p=1:5
    mkdir(strcat(savedatadirname, 'labeled/', 'P', num2str(p), '/', 'cam1') );
    mkdir(strcat(savedatadirname, 'labeled/', 'P', num2str(p), '/', 'cam2') );
end
for p=1:5
    p
    for nump = 1:size(labeled{p}, 1)
        for c = 1:5
            X = labeled{p}{nump, c};
            if size(X,1) == 0
            else
                name = strcat( 'labeled/', 'P', num2str(p), '/', 'cam1/', num2str(nump), '-', num2str(c), '.png');
                imwrite(X, strcat(savedatadirname, name));
            end
        end
        for c=6:10
            X = labeled{p}{nump, c};
            if size(X,1) == 0
            else
                name = strcat( 'labeled/', 'P', num2str(p), '/', 'cam2/', num2str(nump), '-', num2str(c), '.png');
                imwrite(X, strcat(savedatadirname, name));
            end
        end
    end
end

% detected
disp('detected');
for p=1:5
    mkdir(strcat(savedatadirname, 'detected/', 'P', num2str(p), '/', 'cam1') );
    mkdir(strcat(savedatadirname, 'detected/', 'P', num2str(p), '/', 'cam2') );
end
for p=1:5
    p
    for nump = 1:size(detected{p}, 1)
        for c = 1:5
            X = detected{p}{nump, c};
            if size(X,1) == 0
            else
                name = strcat( 'detected/', 'P', num2str(p), '/', 'cam1/', num2str(nump), '-', num2str(c), '.png');
                imwrite(X, strcat(savedatadirname, name));
            end
        end
        for c=6:10
            X = detected{p}{nump, c};
            if size(X,1) == 0
            else
                name = strcat( 'detected/', 'P', num2str(p), '/', 'cam2/', num2str(nump), '-', num2str(c), '.png');
                imwrite(X, strcat(savedatadirname, name));
            end
        end
    end
end


