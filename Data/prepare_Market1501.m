clc
clear 
%% PLEASE CHANGE THE DATASET FOLDER NAME TO Market1501
marketFolder = './Market1501';

%% training
mkdir(fullfile(marketFolder,'train'))
tmp = dir(fullfile(marketFolder,'bounding_box_train/*.jpg'));
names = {tmp.name};
disp('Preparing training...');
for n = 1:numel(names)
    tmpname = names{n};
    if strcmp(tmpname(1),'-')
        gID = 9999; %use 9999 for "junk"
        camID = str2num(tmpname(5));
    else
        gID = str2num(tmpname(1:4));
        camID = str2num(tmpname(7));
    end
    tmppath = fullfile(marketFolder,sprintf('train/cam%d/%04d',camID,gID));
    if exist(tmppath,'dir')==0
        mkdir(tmppath)
    end
    movefile(fullfile(marketFolder,'bounding_box_train',tmpname),fullfile(tmppath,tmpname));
%     im = imread(fullfile(marketFolder,'bounding_box_train',tmpname));
%     imwrite(im,fullfile(tmppath,tmpname));
end

%% testing-gallery
mkdir(fullfile(marketFolder,'test/gallery'))
tmp = dir(fullfile(marketFolder,'bounding_box_test/*.jpg'));
names = {tmp.name};
disp('Preparing gallery...');
for n = 1:numel(names)
    tmpname = names{n};
    if strcmp(tmpname(1),'-')
        gID = 9999; %use 9999 for "junk"
        camID = str2num(tmpname(5));
    else
        gID = str2num(tmpname(1:4));
        camID = str2num(tmpname(7));
    end
    tmppath = fullfile(marketFolder,sprintf('test/gallery/cam%d/%04d',camID,gID));
    if exist(tmppath,'dir')==0
        mkdir(tmppath)
    end
    movefile(fullfile(marketFolder,'bounding_box_test',tmpname),fullfile(tmppath,tmpname));
%     im = imread(fullfile(marketFolder,'bounding_box_test',tmpname));
%     imwrite(im,fullfile(tmppath,tmpname));
end

%% testing-probe
mkdir(fullfile(marketFolder,'test/probe'))
tmp = dir(fullfile(marketFolder,'query/*.jpg'));
names = {tmp.name};
disp('Preparing probe...');
for n = 1:numel(names)
    tmpname = names{n};
    if strcmp(tmpname(1),'-')
        gID = 9999; %use 9999 for "junk"
        camID = str2num(tmpname(5));
    else
        gID = str2num(tmpname(1:4));
        camID = str2num(tmpname(7));
    end
    tmppath = fullfile(marketFolder,sprintf('test/probe/cam%d/%04d',camID,gID));
    if exist(tmppath,'dir')==0
        mkdir(tmppath)
    end
    movefile(fullfile(marketFolder,'query',tmpname),fullfile(tmppath,tmpname));
%     im = imread(fullfile(marketFolder,'query',tmpname));
%     imwrite(im,fullfile(tmppath,tmpname));
end