%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% set_database.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
datadirname_root = 'E:/database_reid/';  % indicate path that contains your dataset
featuredirname_root = 'Features/'; % indicate root path to save features

switch sys.database
    case 1 
        databasename = 'VIPeR';
        datadirname = strcat(datadirname_root, databasename, '/');
        featuredirname = featuredirname_root;
        DBfile = 'DB/VIPeR.mat';
        
        numperson_train = 316;
        numperson_probe = 316;
        numperson_garalley = 316;
    case 2 
        databasename = 'CUHK01'; % CUHK01(M=1) singleshot 
        datadirname = strcat(datadirname_root, 'CUHK01/campus/');
        featuredirname = featuredirname_root;
        DBfile = 'DB/CUHK01M1.mat';
        
        % person number is same as [25] (Paisitkriangkrai et. al 2015)
        numperson_train = 486;
        numperson_probe = 485;
        numperson_garalley = 485;
    case 3 
        databasename = 'CUHK01';  % CUHK01(M=2) multishot
        datadirname = strcat(datadirname_root, 'CUHK01/campus/');
        featuredirname = featuredirname_root;
        DBfile = 'DB/CUHK01M2.mat';
        
        % person number is same as [26] (Liao et. al 2015) 
        numperson_train = 485;
        numperson_probe = 486;
        numperson_garalley = 486;
    case 4
        databasename = 'PRID450s';
        datadirname = strcat(datadirname_root, '/prid_450s/');
        featuredirname = featuredirname_root;
        DBfile = 'DB/PRID450s.mat';
        
        numperson_train = 225;
        numperson_garalley = 225;
        numperson_probe = 225;
    case 5
        databasename = 'GRID';
        datadirname = strcat(datadirname_root, databasename, '/');
        featuredirname = featuredirname_root;
        DBfile = 'DB/GRID.mat';
          
        numperson_train = 125;
        numperson_probe = 125;
        numperson_garalley = 900;
    case 6 
        databasename = 'CUHK03labeled';
        datadirname = strcat(datadirname_root, 'CUHK03/labeled/');
        featuredirname = featuredirname_root;
        DBfile = 'DB/CUHK03labeled.mat';
       
        numperson_train = 1260;
        numperson_garalley = 100;
        numperson_probe = 100;
    case 7 
        databasename = 'CUHK03detected';
        datadirname = strcat(datadirname_root, 'CUHK03/detected/');
        featuredirname = featuredirname_root;
        DBfile = 'DB/CUHK03detected.mat';
        
        numperson_train = 1260;
        numperson_garalley = 100;
        numperson_probe = 100;
    otherwise
        fprintf('lf_type = %d is not defined', lf_type);
end
load(DBfile,  'allimagenames', 'traininds_set', 'testinds_set', ...
    'trainimagenames_set', 'testimagenames_set', 'trainlabels_set', ...
    'testlabels_set', 'traincamIDs_set', 'testcamIDs_set' );

allimagenums = size(allimagenames, 1);






