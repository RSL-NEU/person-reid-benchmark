function [ imgs,camID,personID ] = parsingDataset( dopts,partition )
%[ imgs,camID,personID ] = parsingDataset( dopts,partition )
%   read all images from the dataset and parse the IDs
    
fprintf('Start parsing dataset...');
switch lower(dopts.name)
    case 'viper'
        [srcdir,cam_sets,num_cam] = folderList(fullfile(dopts.datafolder, 'VIPeR'));
        camID = [];
        personID = [];
        imgs = {};
        cnt = 1;
        for c = 1:num_cam
            [~,person_sets,num_person] = folderList(fullfile(srcdir,cam_sets{c}));
            for p = 1:num_person
                iminfo = dir(fullfile(srcdir,cam_sets{c},person_sets{p},'*.bmp'));
                for i = 1:numel(iminfo)
                    imgs{cnt} = imread(fullfile(srcdir,cam_sets{c},person_sets{p},...
                        iminfo(i).name));
                    camID(cnt) = str2num(cam_sets{c}(end));
                    personID(cnt) = str2num(person_sets{p});
                    cnt = cnt + 1;
                end
            end
        end           
    case 'msmt17'
        tmp_path = pwd;
        imsz = [128,64]; % resize image
        cd(fullfile(dopts.datafolder, 'MSMT17_V1'));
        camID = zeros(1,numel(partition.idx_train));
        personID = zeros(1,numel(partition.idx_train));
        imgs = cell(1,numel(partition.idx_train));
        cnt = 1;
        % training
        fid = fopen('list_train.txt');
        tmpline = fgetl(fid);
        cnt = 1;

        while 1
            if tmpline == -1 
                break;
            end
            disp(tmpline)
            personID(cnt) = str2double(tmpline(1:4));
            camID(cnt) = str2double(tmpline(15:16));
            filename = strsplit(tmpline);
            imgs{cnt} = imresize(imread(fullfile('train',filename{1})),imsz);
            cnt = cnt + 1;   
            tmpline = fgetl(fid);
        end
        fclose(fid);

        % validating
        fid = fopen('list_val.txt');
        tmpline = fgetl(fid);        
        while 1
            if tmpline == -1 
                break;
            end
            disp(tmpline)
            personID(cnt) = str2double(tmpline(1:4));
            camID(cnt) = str2double(tmpline(15:16));
            filename = strsplit(tmpline);
            imgs{cnt} = imresize(imread(fullfile('train',filename{1})),imsz);
            cnt = cnt + 1;   
            tmpline = fgetl(fid);
        end
        fclose(fid);

        % query
        ID_offset = max(personID)+1; % make sure the ID is unique for both train/test
        fid = fopen('list_query.txt');
        tmpline = fgetl(fid);
        while 1
            if tmpline == -1 
                break;
            end
            disp(tmpline)
            personID(cnt) = str2double(tmpline(1:4)) + ID_offset;
            camID(cnt) = str2double(tmpline(15:16));
            filename = strsplit(tmpline);
            imgs{cnt} = imresize(imread(fullfile('test',filename{1})),imsz);
            cnt = cnt + 1;   
            tmpline = fgetl(fid);
        end
        fclose(fid);

        % gallery
        fid = fopen('list_gallery.txt');
        tmpline = fgetl(fid);
        while 1
            if tmpline == -1 
                break;
            end
            disp(tmpline)
            personID(cnt) = str2double(tmpline(1:4)) + ID_offset;
            camID(cnt) = str2double(tmpline(15:16));
            filename = strsplit(tmpline);
            imgs{cnt} = imresize(imread(fullfile('test',filename{1})),imsz);
            cnt = cnt + 1;   
            tmpline = fgetl(fid);
        end
        fclose(fid);
        cd(tmp_path);
    case 'dukemtmc'        
        folder = fullfile(dopts.datafolder, 'DukeMTMC4ReID','ReID');
        [~,caminfo,] = folderList(folder);
        imgs = {};
        personID = [];
        camID = [];
        impath = {};
        cnt = 1;
        for c = 1:numel(caminfo)
            fprintf('Parsing camera %d...\n',c);
            camID_tmp = str2num(caminfo{c}(end));
            [~,persons,~] = folderList(fullfile(folder,caminfo{c}));
            for p = 1:numel(persons)
                iminfo = dir(fullfile(folder,caminfo{c},persons{p},'*.jpg'));
                for i = 1:numel(iminfo)
                    impath_tmp = fullfile(folder,caminfo{c},persons{p},iminfo(i).name);
                    tmpim = imread(impath_tmp);
                    imgs{cnt} = tmpim;
                    personID(cnt) = str2num(persons{p});
                    camID(cnt) = camID_tmp;
                    impath{cnt} = impath_tmp;
                    cnt = cnt + 1;
                end
            end
        end
        imgs = cellfun(@(x) imresize(x,[128 64]),imgs,'uni',0);        
    case 'cuhk01'
        folder = fullfile(dopts.datafolder,'campus');
        load(fullfile(folder,'CUHK01.mat'));
        personID = [];
        imgs = {};
        camID = [];
        for i = 1:numel(allimagenames)
            tmp = imread(fullfile(folder,allimagenames{i}));
            imgs{i} = imresize(tmp,[128 48]);
            personID(i) = str2num(allimagenames{i}(2:5));
            camID(i) = ceil(str2num(allimagenames{i}(6:8))./2);
        end       
    case 'caviar'
        [srcdir,cam_sets,num_cam] = folderList(fullfile(dopts.datafolder,'CAVIARa'));
        camID = [];
        personID = [];
        imgs = {};
        cnt = 1;
        for c = 1:num_cam
            [~,person_sets,num_person] = folderList(fullfile(srcdir,cam_sets{c}));
            for p = 1:num_person
                iminfo = dir(fullfile(srcdir,cam_sets{c},person_sets{p},'*.jpg'));
                for i = 1:numel(iminfo)
                    imgs{cnt} = imread(fullfile(srcdir,cam_sets{c},person_sets{p},...
                        iminfo(i).name));
                    imgs{cnt}=imresize(imgs{cnt},[128 48]);
                    camID(cnt) = str2num(cam_sets{c}(end));
                    personID(cnt) = str2num(person_sets{p});
                    cnt = cnt + 1;
                end
            end
        end                
    case '3dpes'
        [srcdir,cam_sets,num_cam] = folderList(fullfile(dopts.datafolder,'3DPeS/parsed3DPeS'));
        camID = [];
        personID = [];
        imgs = {};
        cnt = 1;
        [~,person_sets,num_person] = folderList(fullfile(srcdir,cam_sets{1}));
        for p = 1:num_person
            iminfo = dir(fullfile(srcdir,cam_sets{1},person_sets{p},'*.bmp'));
            for i = 1:numel(iminfo)
                imgs{cnt} = imread(fullfile(srcdir,cam_sets{1},person_sets{p},...
                    iminfo(i).name));
                imgs{cnt}=imresize(imgs{cnt},[128 48]);
                camID(cnt) = str2num(cam_sets{1}(end));
                personID(cnt) = str2num(person_sets{p}(7:end));
                cnt = cnt + 1;
            end
        end
        [n,x] = hist(personID,unique(personID));
        id_inval = x(n==1);        
        camID(ismember(personID,id_inval)) = [];
        imgs(ismember(personID,id_inval)) = [];
        personID(ismember(personID,id_inval)) = [];
        
    case 'hda'
        load(fullfile(dopts.datafolder,'HDA_Dataset_V1.3/hda_t60_OF_off.mat'));
    case 'grid'
        % images are normalized to 128x48
        [srcdir,cam_sets,num_cam] = folderList(fullfile(dopts.datafolder,'QMUL_GRID'));
        camID = [];
        personID = [];
        imgs = {};
        cnt = 1;
        for c = 1:2%num_cam
            [~,person_sets,num_person] = folderList(fullfile(srcdir,cam_sets{c}));
            for p = 1:num_person
                iminfo = dir(fullfile(srcdir,cam_sets{c},person_sets{p},'*.jpeg'));
                for i = 1:numel(iminfo)
                    tmpim = imread(fullfile(srcdir,cam_sets{c},person_sets{p},...
                        iminfo(i).name));
                    imgs{cnt} = imresize(tmpim,[128 48]);
                    camID(cnt) = str2num(cam_sets{c}(end));
                    personID(cnt) = str2num(person_sets{p});
                    cnt = cnt + 1;
                end
            end
        end        
    case 'market'
        % parsing training part 
        [srcdir,cam_sets,num_cam] = folderList(fullfile(dopts.datafolder,'Market1501/train'));
        camID = [];
        personID = [];
        imgs = {};        
        cnt = 1;
        for c = 1:num_cam
            [~,person_sets,num_person] = folderList(fullfile(srcdir,cam_sets{c}));
            for p = 1:num_person
                iminfo = dir(fullfile(srcdir,cam_sets{c},person_sets{p},'*.jpg'));
                for i = 1:numel(iminfo)
                    imgs{cnt} = imread(fullfile(srcdir,cam_sets{c},person_sets{p},...
                        iminfo(i).name));
                    camID(cnt) = str2num(cam_sets{c}(end));
                    personID(cnt) = str2num(person_sets{p});
                    cnt = cnt + 1;
                end
            end
        end
        
        num_train = cnt-1;
        % parsing testing part - probe
        [srcdir,cam_sets,num_cam] = folderList(fullfile(dopts.datafolder,'Market1501/test/probe'));
        for c = 1:num_cam
            [~,person_sets,num_person] = folderList(fullfile(srcdir,cam_sets{c}));
            for p = 1:num_person
                iminfo = dir(fullfile(srcdir,cam_sets{c},person_sets{p},'*.jpg'));
                for i = 1:numel(iminfo)
                    imgs{cnt} = imread(fullfile(srcdir,cam_sets{c},person_sets{p},...
                        iminfo(i).name));
                    camID(cnt) = str2num(cam_sets{c}(end));
                    personID(cnt) = str2num(person_sets{p});
                    cnt = cnt + 1;
                end
            end
        end
        num_probe = cnt - 1 - num_train;
        
        % parsing testing part - gallery
        [srcdir,cam_sets,num_cam] = folderList(fullfile(dopts.datafolder,'Market1501/test/gallery'));
        for c = 1:num_cam
            [~,person_sets,num_person] = folderList(fullfile(srcdir,cam_sets{c}));
            for p = 1:num_person
                iminfo = dir(fullfile(srcdir,cam_sets{c},person_sets{p},'*.jpg'));
                for i = 1:numel(iminfo)
                    if str2num(person_sets{p}) == 9999 % ignore junk image
                        continue;
                    end
                    imgs{cnt} = imread(fullfile(srcdir,cam_sets{c},person_sets{p},...
                        iminfo(i).name));
                    camID(cnt) = str2num(cam_sets{c}(end));
                    personID(cnt) = str2num(person_sets{p});
                    cnt = cnt + 1;
                end
            end
        end
        
        trainIdx = zeros(1,numel(personID));
        trainIdx(1:num_train) = 1;
        probeIdx = zeros(1,numel(personID)-num_train);
        probeIdx(1:num_probe) = 1;
      
        method.trainIdx = logical(trainIdx);
        method.probeIdx = logical(probeIdx);
    case 'cuhk02' 
        % images are normalized to 128x64
        % personID 3012 means the 12th identity in camera pair 3
        [srcdir,cam_pairs,num_pair] = folderList(fullfile(dopts.datafolder,'cuhk02'));
        camID = [];
        personID = [];
        imgs = {};        
        cnt = 1;
        for cpair = 1:5 % only use the first 3 pairs
            [~,cam_sets,num_cam] = folderList(fullfile(srcdir,cam_pairs{cpair}));
            for c = 1:num_cam
                [~,person_sets,num_person] = folderList(fullfile(srcdir,cam_pairs{cpair},cam_sets{c}));
                for p = 1:num_person
                    iminfo = dir(fullfile(srcdir,cam_pairs{cpair},cam_sets{c},person_sets{p},'*.png'));
                    for i = 1:numel(iminfo)
                        tmpim = imread(fullfile(srcdir,cam_pairs{cpair},cam_sets{c},person_sets{p},...
                            iminfo(i).name));
                        imgs{cnt} = imresize(tmpim,[128 64]);
                        camID(cnt) = str2num(cam_sets{c}(end))+10*cpair;
                        personID(cnt) = str2num(person_sets{p})+1000*cpair;
                        cnt = cnt + 1;
                    end
                end
            end
        end        
    case 'cuhk_detected' 
        % images are normalized to 128x64
        % personID 3012 means the 12th identity in camera pair 3
        [srcdir,cam_pairs,num_pair] = folderList(fullfile(dopts.datafolder,'CUHK03/detected'));
        camID = [];
        personID = [];
        imgs = {};        
        cnt = 1;
        for cpair = 1:3 % only use the first 3 pairs
            [~,cam_sets,num_cam] = folderList(fullfile(srcdir,cam_pairs{cpair}));
            for c = 1:num_cam
                [~,person_sets,num_person] = folderList(fullfile(srcdir,cam_pairs{cpair},cam_sets{c}));
                for p = 1:num_person
                    iminfo = dir(fullfile(srcdir,cam_pairs{cpair},cam_sets{c},person_sets{p},'*.png'));
                    for i = 1:numel(iminfo)
                        tmpim = imread(fullfile(srcdir,cam_pairs{cpair},cam_sets{c},person_sets{p},...
                            iminfo(i).name));
                        imgs{cnt} = imresize(tmpim,[128 64]);
                        camID(cnt) = str2num(cam_sets{c}(end))+10*cpair;
                        personID(cnt) = str2num(person_sets{p})+1000*cpair;
                        cnt = cnt + 1;
                    end
                end
            end
        end
        
    case 'cuhk_labeled'
        % images are normalized to 128x64
        % personID 3012 means the 12th identity in camera pair 3
        [srcdir,cam_pairs,num_pair] = folderList(fullfile(dopts.datafolder,'CUHK03/labeled'));
        camID = [];
        personID = [];
        imgs = {};        
        cnt = 1;
        for pair = 1:3 % only use the first 3 pairs
            [~,cam_sets,num_cam] = folderList(fullfile(srcdir,cam_pairs{pair}));
            for c = 1:num_cam
                [~,person_sets,num_person] = folderList(fullfile(srcdir,cam_pairs{pair},cam_sets{c}));
                for p = 1:num_person
                    iminfo = dir(fullfile(srcdir,cam_pairs{pair},cam_sets{c},person_sets{p},'*.png'));
                    for i = 1:numel(iminfo)
                        tmpim = imread(fullfile(srcdir,cam_pairs{pair},cam_sets{c},person_sets{p},...
                            iminfo(i).name));
                        imgs{cnt} = imresize(tmpim,[128 64]);
                        camID(cnt) = str2num(cam_sets{c}(end))+10*pair;
                        personID(cnt) = str2num(person_sets{p})+1000*pair;
                        cnt = cnt + 1;
                    end
                end
            end
        end
    case 'ilidsvid'
        [srcdir,cam_sets,num_cam] = folderList(fullfile(dopts.datafolder,'i-LIDS-VID/multi_shot'));
        camID = [];
        personID = [];
        imgs = {};        
        cnt = 1;
        for c = 1:num_cam
            [~,person_sets,num_person] = folderList(fullfile(srcdir,cam_sets{c}));
            for p = 1:num_person
                iminfo = dir(fullfile(srcdir,cam_sets{c},person_sets{p},'*.png'));
                for i = 1:numel(iminfo)
                    imgs{cnt} = imread(fullfile(srcdir,cam_sets{c},person_sets{p},...
                        iminfo(i).name));
                    camID(cnt) = str2num(cam_sets{c}(end));
                    personID(cnt) = str2num(person_sets{p});
                    cnt = cnt + 1;
                end
            end
        end              
        
    case 'v47'
        [srcdir,cam_sets,num_cam] = folderList(fullfile(dopts.datafolder,'V-47_Images/StandardFormat'));
        camID = [];
        personID = [];
        imgs = {};        
        cnt = 1;
        for c = 1:num_cam
            [~,person_sets,num_person] = folderList(fullfile(srcdir,cam_sets{c}));
            for p = 1:num_person
                iminfo = dir(fullfile(srcdir,cam_sets{c},person_sets{p},'*.png'));
                for i = 1:numel(iminfo)
                    tmpim = imread(fullfile(srcdir,cam_sets{c},person_sets{p},...
                        iminfo(i).name));
                    imgs{cnt}=imresize(tmpim,[128 64]);
                    camID(cnt) = str2num(cam_sets{c}(end));
                    personID(cnt) = str2num(person_sets{p});
                    cnt = cnt + 1;
                end
            end
        end
                
    case 'saivt'
        if(strcmp(pair,'38'))
            [srcdir,cam_sets,num_cam] = folderList(fullfile(dopts.datafolder,'saivt/38'));
        else
            [srcdir,cam_sets,num_cam] = folderList(fullfile(dopts.datafolder,'saivt/58'));
        end
        camID = [];
        personID = [];
        imgs = {};        
        cnt = 1;
        for c = 1:num_cam
            [~,person_sets,num_person] = folderList(fullfile(srcdir,cam_sets{c}));
            for p = 1:num_person
                iminfo = dir(fullfile(srcdir,cam_sets{c},person_sets{p},'*.jpeg'));
                for i = 1:numel(iminfo)
                    tmpim = imread(fullfile(srcdir,cam_sets{c},person_sets{p},...
                        iminfo(i).name));
                    imgs{cnt} = imresize(tmpim,[128 64]);
                    camID(cnt) = str2num(cam_sets{c}(end));
                    personID(cnt) = str2num(person_sets{p});
                    cnt = cnt + 1;
                end
            end
        end        
        
    case 'ward'
        if(strcmp(pair,'12'))
            [srcdir,cam_sets,num_cam] = folderList(fullfile(dopts.datafolder,'WARD/12'));
        else
            [srcdir,cam_sets,num_cam] = folderList(fullfile(dopts.datafolder,'WARD/13'));
        end
        camID = [];
        personID = [];
        imgs = {};        
        cnt = 1;
        for c = 1:num_cam
            [~,person_sets,num_person] = folderList(fullfile(srcdir,cam_sets{c}));
            for p = 1:num_person
                iminfo = dir(fullfile(srcdir,cam_sets{c},person_sets{p},'*.png'));
                for i = 1:numel(iminfo)
                    tmpim = imread(fullfile(srcdir,cam_sets{c},person_sets{p},...
                        iminfo(i).name));
                    imgs{cnt} = imresize(tmpim,[128 64]);
                    camID(cnt) = str2num(cam_sets{c}(end));
                    personID(cnt) = str2num(person_sets{p});
                    cnt = cnt + 1;
                end
            end
        end
        
    case 'raid'
        if(strcmp(pair,'12'))
            [srcdir,cam_sets,num_cam] = folderList(fullfile(dopts.datafolder,'RAiD/12'));
        elseif(strcmp(pair,'13'))
            [srcdir,cam_sets,num_cam] = folderList(fullfile(dopts.datafolder,'RAiD/13'));
        else
            [srcdir,cam_sets,num_cam] = folderList(fullfile(dopts.datafolder,'RAiD/14'));
        end
        camID = [];
        personID = [];
        imgs = {};        
        cnt = 1;
        for c = 1:num_cam
            [~,person_sets,num_person] = folderList(fullfile(srcdir,cam_sets{c}));
            for p = 1:num_person
                iminfo = dir(fullfile(srcdir,cam_sets{c},person_sets{p},'*.png'));
                for i = 1:numel(iminfo)
                    tmpim = imread(fullfile(srcdir,cam_sets{c},person_sets{p},...
                        iminfo(i).name));
                    imgs{cnt} = imresize(tmpim,[128 64]);
                    camID(cnt) = str2num(cam_sets{c}(end));
                    personID(cnt) = str2num(person_sets{p});
                    cnt = cnt + 1;
                end
            end
        end
        
    case 'airport'
        fid = fopen(fullfile(dopts.datafolder, 'AirportALERT_image','filepath.txt'));
        cnt = 1;
        tline = fgetl(fid);
        tline = strrep(tline,'\','/');
        while ischar(tline)
            imgs{cnt} = imread(fullfile(dopts.datafolder, 'AirportALERT_image',tline));
            cnt = cnt + 1;
            tline = fgetl(fid);
            tline = strrep(tline,'\','/');
        end
        fclose(fid);
        fid = fopen(fullfile(dopts.datafolder, 'AirportALERT_image','camID.txt'));
        camID = fscanf(fid,'%d');
        fclose(fid);
        fid = fopen(fullfile(dopts.datafolder, 'AirportALERT_image','personID.txt'));
        personID = fscanf(fid,'%d');        
        fclose(fid);
        camID = camID';
        personID = personID';
    case 'prid'
        [srcdir,cam_sets,num_cam] = folderList(fullfile(dopts.datafolder,'prid_2011/multi_shot_DVR/'));
        camID = [];
        personID = [];
        imgs = {};        
        cnt = 1;
        for c = 1:num_cam
            [~,person_sets,num_person] = folderList(fullfile(srcdir,cam_sets{c}));
            for p = 1:num_person
                iminfo = dir(fullfile(srcdir,cam_sets{c},person_sets{p},'*.png'));
                for i = 1:numel(iminfo)
                    imgs{cnt} = imread(fullfile(srcdir,cam_sets{c},person_sets{p},...
                        iminfo(i).name));
                    camID(cnt) = str2num(cam_sets{c}(end));
                    personID(cnt) = str2num(person_sets{p}(end-2:end));
                    cnt = cnt + 1;
                end
            end
        end
        
end
% display statistics
disp('Done!')
disp('##################################')
fprintf('Dataset---%s\n',dopts.name);
fprintf('# images---%d\n',numel(imgs));
fprintf('# persons---%d\n',numel(unique(personID)));
fprintf('# cameras---%d\n',numel(unique(camID)));
disp('##################################')

end

