%%%%%
% Main script to compute features
% Inputs: img: {MxNx3} image(s) to compute features
%         method: structure to keep the parameter 
%           .name: 'gbicov','ldfv','sdc','color_texture', 'hist_LBP',
%           'lomo','gog'
%           .idx_train: training index
% Output: x: feature vector
%         method: parameters to keep
%%%%%

function [x,method]=ComputeFeatures(img,method)
warning off

imsz = size(img{1});
patches = method.patchSize;
grids = method.stepSize;

if ~iscell(img)
    img = {img};
end

switch lower(method.featureType)
    case 'whos'
        patchsize = patches;%[imsz(2) 20]; 
        gridstep = grids;%[imsz(2) 20];
        imsz = size(img{1});
        imsz = imsz(1:2);
        
        [kx,ky] = meshgrid(1:imsz(2),1:imsz(1));
        % centering
        kx = (kx - imsz(2)/2).^2;
        ky = (ky - imsz(1)/2).^2;
        sigma = (imsz/4).^2;
        K_no_iso_gaus = exp(-(kx./sigma(2))-(ky./sigma(1)));
        fprintf('Begin to extract WHOS feature');
        for i = 1:numel(img)
            if mod(i,round(numel(img)/10))==0
                fprintf('.');
            end               
            x(i,:) = PETA_cal_img_full_hist(img{i},K_no_iso_gaus,1,patchsize,gridstep);
        end
        fprintf('Done!\n')
        fprintf('# dimensions --- %d\n',size(x,2));
    case 'gog'
        param=set_default_parameter(1);
        param.lfparam.usebase = [1 1 1 0 0 0];
        param.lfparam.num_element = 8;
        param.lfparam.lf_name = 'yMRGB';
        param.d = 8;
        param.G = method.numPatch; % 6 strip
        param.patchSize = floor(patches);
        param.gridSize = floor(grids);        
        method.option = param;
        
        fprintf('Begin to extract GOG feature');
        x=GOGHelper(img,param);
        fprintf('Done!\n')
        fprintf('# dimensions --- %d\n',size(x,2));
    case 'gbicov'
        % default setting: size - [16 16]; step - [4 4]
        % 6 strips - [48 20]
        patchsize = floor(patches); %patchs;%[imsz(2) 20]; 
        gridstep = floor(grids); %grids;%[imsz(2) 20];

        fprintf('Begin to extract gbicov feature...');
        x = gbicovEx(img,patchsize,gridstep);
        x = x';   
        fprintf('Done!\n')
        fprintf('# dimensions --- %d\n',size(x,2));
    case 'ldfv'
        nNode = 16;
        method.option.nNode = nNode;
        
        patchsize = floor(patches); %patchs;%[imsz(2) 20]; 
        gridstep = floor(grids); %grids;%[imsz(2) 20];                
        method.numPatch = floor(((size(img{1},2) - patchsize(1))/gridstep(1)+1))...
                        * floor(((size(img{1},1) - patchsize(2))/gridstep(2)+1));
        
        train = find(method.idx_train);    
        fprintf('Begin to extract LDFV feature...');
        x = LDFV(img,train, patchsize,gridstep,nNode);
        fprintf('Done!\n')
        fprintf('# dimensions --- %d\n',size(x,2));
    case 'sdc'
        patchsize = patches;%[imsz(2) 128/6]; %[10 10];%
        gridstep = grids;%[imsz(2) 128/6]; %[4 4];%
        
        fprintf('Begin to extract DenseColorSIFT feature...');
        x = dense_colorsift(img,patchsize,gridstep);
        x = x';
        fprintf('Done!\n')
        fprintf('# dimensions --- %d\n',size(x,2));
    case 'color_texture'
        nBins=16;        
        method.options.nBins = nBins;
        
        fprintf('Begin to extract ELF feature...');
        x=colorTexture(img,method.numRow,nBins);
        fprintf('Done!\n')
        fprintf('# dimensions --- %d\n',size(x,2));        
    case 'hist_lbp'
        nBins=16;        
        method.options.nBins = nBins;
        
        patchsize = patches;%[imsz(2) 128/6];
        gridstep = grids;%[imsz(2) 128/6];       
        fprintf('Begin to extract HistLBP feature...');
        x=HistLBP(img,nBins,patchsize,gridstep);
        fprintf('Done!\n')
        fprintf('# dimensions --- %d\n',size(x,2));      
    case 'lomo'
        options.numScales = 1;
        options.blockSize = round(min(patches)); %patchs(1);%
        options.blockStep = round(min(patches)); % grids(1);
        options.hsvBins = [8,8,8];
        options.tau = 0.3;
        options.R = [3, 5];
        options.numPoints = 4;
        method.options = options;
        
        [h,w,~] = size(img{1});
        images = zeros(h, w, 3, numel(img), 'uint8');
        for i = 1 : numel(img)
            images(:,:,:,i) = img{i};
        end
        fprintf('Begin to extract LOMO feature...');
        x = LOMO(images, options);
        x = x';
        fprintf('Done!\n')
        fprintf('# dimensions --- %d\n',size(x,2));      
    case 'DEEP'
        if ~isfield(method,'model')
            method.model = 'alex';
        end
        switch method.model
            case 'alex'
                net = load('../Misc/matconvnet/imagenet-caffe-alex.mat');
                net.layers(end-1:end) = []; % remove the last classification layer
            case 'vgg19'
                net = load('../Misc/matconvnet/imagenet-vgg-verydeep-19.mat');
                net.layers(end-1:end) = []; % remove the last classification layer        
            case 'siamens'
                net = load(fullfile('../NeuralNet/trained',['net_' method.data '.mat']));
                net.layers(end) = []; % remove the last classification layer
            otherwise 
                error('Unrecognized model name!')
        end
        vl_simplenn_display(net);
        batchsize = 50;
        method.batchsize = batchsize;
        method.name = method.model;
        x = deepfeat(img,net,batchsize);
        x = x';
     case 'IDE_ResNet'
        modelFile='D:\benchmark\code\NeuralNet\trainedModels\all\1\train_test_solver\test.prototxt';
        %currSplitMeanFile='D:\benchmark\code\NeuralNet\trainedModels\Market-airport-duke\1\mean_binaryprotos\NonSiamese\mean_market_airport_duke.png';
        %meanImg=imread(currSplitMeanFile);    
        % Read mean image - note, this is triplet mean        
        %meanImg=meanImg(:,:,[3, 2, 1]);
        %meanImg=single(permute(meanImg,[2, 1, 3]));
        
        % Preprocess images
        img1={};
        for i=1:numel(img)
            % Resize
            if((size(img{i},1)~=224) || (size(img{i},2)~=224))
                img1{i}=imresize(img{i},[224,224]);
            else
                img1{i}=img{i};
            end
            % Convert RGB to BGR
            if(size(img{i},3)==3)
                img1{i} = img1{i}(:, :, [3, 2, 1]);
            end
            img1{i}=single(img1{i});
            % Subtract mean
            img1{i}(:,:,3)=img1{i}(:,:,3)-104;
            img1{i}(:,:,2)=img1{i}(:,:,2)-117;
            img1{i}(:,:,1)=img1{i}(:,:,1)-124;
            % Flip width and height, convert to single
            img1{i} = permute(img1{i},[2, 1, 3]);
        end
        
        for sp=1:10
            caffe.set_mode_gpu();
            caffe.set_device(2); 
            networkFile=strcat('D:\benchmark\code\NeuralNet\trainedModels\all\',num2str(sp),'\model\train_iter_75000.caffemodel');
            net=caffe.Net(modelFile, networkFile, 'test');
            x{sp}=computeFeaturesCNN(img1,[],net);
            %x=computeFeaturesCNN(img1,[],net);
            caffe.reset_all();
        end
     case 'BaseNet'
        
        nSplits=1;
        
        % Preprocess images
        img1={};
        for i=1:numel(img)
            % Resize
            if((size(img{i},1)~=100) || (size(img{i},2)~=100))
                img1{i}=imresize(img{i},[100,100]);
            else
                img1{i}=img{i};
            end
            % Convert RGB to BGR
            if(size(img{i},3)==3)
                img1{i} = img1{i}(:, :, [3, 2, 1]);
            end
            % Flip width and height, convert to single
            img1{i} = single(permute(img1{i},[2, 1 3])); 
        end

        caffe.set_mode_gpu();
        caffe.set_device(0);
        
        for i=1:nSplits
            currSplitModelFile='E:\Dropbox\Research\ConvolutionalNets\Re-Id\trainedNetworks\benchmarkSplits\1\Siamese\BaseNet\trainprid10k\config\deploy.prototxt';
            currSplitNetworkFile='E:\Dropbox\Research\ConvolutionalNets\Re-Id\trainedNetworks\benchmarkSplits\1\Siamese\BaseNet\trainprid10k\model\train_iter_10000.caffemodel';
            currSplitMeanFile='E:\Dropbox\Research\ConvolutionalNets\Re-Id\data_LevelDB\benchmarkSplits\1\mean_binaryprotos\mean_prid10k.png';
            
            % Read mean image - note, this is triplet mean
            meanImg=imresize(imread(currSplitMeanFile),[100 100]);
            meanImg=meanImg(:,:,[3, 2, 1]);
            meanImg=permute(meanImg,[2, 1, 3]);
%             mean_data=caffe.io.read_mean(currSplitMeanFile);
%             meanImg=[];
%             for j=1:3
%                 meanImg(:,:,j)=(1/3)*(mean_data(:,:,j)+mean_data(:,:,j+3)+mean_data(:,:,j+6));
%             end
%             for j=1:3
%                 meanImg(:,:,j)=(1/2)*(mean_data(:,:,j)+mean_data(:,:,j+3));
%             end
            % Initialize network
            net=caffe.Net(currSplitModelFile, currSplitNetworkFile, 'test');
            x=computeFeaturesCNN(img1,single(meanImg),net);
        end
end
warning on
end

