function [idx_train,idx_test,idx_probe,idx_gallery] = getTrainTestSplit(personID,camID,data,s)

if(isempty(data.pair))
    load(strcat('Split_',data.name));
else
    load(strcat('Split_',data.name,'_',data.pair));
end
switch data.name
    case 'viper'
        tSize = 316;
        tmpSplit = split(s,:);
        id_train = tmpSplit(1:tSize); % first half to train
        id_train = personID(id_train);
        id_test = tmpSplit(tSize+1:end); 
        id_test = personID(id_test);
        idx_train = ismember(personID,id_train);
        idx_test = ismember(personID,id_test);
        camID_test = camID(idx_test);
        uCamID_test = unique(camID_test);
        for u = 1:numel(uCamID_test)
            idx_gallery(u,:) = camID_test == uCamID_test(u);
            idx_probe(u,:) = ~idx_gallery(u,:);
        end
    case 'cuhk01'
        tSize = 486;
        tmpSplit = split(s,:);
        id_train = tmpSplit(1:numel(tmpSplit)-tSize);
        id_test = tmpSplit(end-tSize+1:end);
        idx_train = ismember(personID,id_train);
        idx_test = ismember(personID,id_test);
        camID_test = camID(idx_test);
        camID_test = mod(camID_test,10);
        uCamID_test = unique(camID_test);
        personID_test = personID(idx_test);
        for u = 1:numel(uCamID_test)
            idx_probe(u,:) = camID_test == uCamID_test(u); % all images in one camera as probe
            idx_gal_tmp = find(~idx_probe(u,:));
            personID_gal_tmp = personID_test(idx_gal_tmp);
            upID_gal_tmp = unique(personID_gal_tmp);
            idx_gal = zeros(1,numel(upID_gal_tmp));
            for up = 1:numel(upID_gal_tmp)
                tmpid = find(personID_gal_tmp==upID_gal_tmp(up));
                idx_gal(up) = tmpid(randsample(numel(tmpid),1));
            end
            idx_gal = idx_gal_tmp(idx_gal);
            idx_gallery(u,:) = ismember(1:numel(personID_test),idx_gal);
        end
    case 'cuhk02'
        tSize = 908;
        tmpSplit = split(s,:);
        id_train = tmpSplit(1:numel(tmpSplit)-tSize);
        id_test = tmpSplit(end-tSize+1:end);
        idx_train = ismember(personID,id_train);
        idx_test = ismember(personID,id_test);
        camID_test = camID(idx_test);
        camID_test = mod(camID_test,10);
        uCamID_test = unique(camID_test);
        personID_test = personID(idx_test);
        for u = 1:numel(uCamID_test)
            idx_probe(u,:) = camID_test == uCamID_test(u); % all images in one camera as probe
            idx_gal_tmp = find(~idx_probe(u,:));
            personID_gal_tmp = personID_test(idx_gal_tmp);
            upID_gal_tmp = unique(personID_gal_tmp);
            idx_gal = zeros(1,numel(upID_gal_tmp));
            for up = 1:numel(upID_gal_tmp)
                tmpid = find(personID_gal_tmp==upID_gal_tmp(up));
                idx_gal(up) = tmpid(randsample(numel(tmpid),1));
            end
            idx_gal = idx_gal_tmp(idx_gal);
            idx_gallery(u,:) = ismember(1:numel(personID_test),idx_gal);
        end
    case 'grid'
        tSize = 125;
        tmpSplit = split(s,:);
        id_train = tmpSplit(1:tSize); % first half to train
        id_train = personID(id_train);
        id_test = tmpSplit(tSize+1:end); 
        id_test = [personID(id_test) 0]; % include distractors
        idx_train = ismember(personID,id_train);
        idx_test = ismember(personID,id_test);
        camID_test = camID(idx_test);
        idx_gallery = camID_test == 2;
        idx_probe = camID_test == 1; 
    case 'cuhk_detected'
        tSize = 100;
        tmpSplit = split(s,:);
        id_train = tmpSplit(1:numel(tmpSplit)-tSize);
        id_test = tmpSplit(end-tSize+1:end);
        idx_train = ismember(personID,id_train);
        idx_test = ismember(personID,id_test);
        camID_test = camID(idx_test);
        camID_test = mod(camID_test,10);
        uCamID_test = unique(camID_test);
        personID_test = personID(idx_test);
        for u = 1:numel(uCamID_test)
            idx_probe(u,:) = camID_test == uCamID_test(u); % all images in one camera as probe
            idx_gal_tmp = find(~idx_probe(u,:));
            personID_gal_tmp = personID_test(idx_gal_tmp);
            upID_gal_tmp = unique(personID_gal_tmp);
            idx_gal = zeros(1,numel(upID_gal_tmp));
            for up = 1:numel(upID_gal_tmp)
                tmpid = find(personID_gal_tmp==upID_gal_tmp(up));
                idx_gal(up) = tmpid(randsample(numel(tmpid),1));
            end
            idx_gal = idx_gal_tmp(idx_gal);
            idx_gallery(u,:) = ismember(1:numel(personID_test),idx_gal);
        end
    case 'cuhk_labeled'
        tSize = 100;
        tmpSplit = split(s,:);
        id_train = tmpSplit(1:numel(tmpSplit)-tSize);
        id_test = tmpSplit(end-tSize+1:end);
        idx_train = ismember(personID,id_train);
        idx_test = ismember(personID,id_test);
        camID_test = camID(idx_test);
        camID_test = mod(camID_test,10);
        uCamID_test = unique(camID_test);
        personID_test = personID(idx_test);
        for u = 1:numel(uCamID_test)
            idx_probe(u,:) = camID_test == uCamID_test(u); % all images in one camera as probe
            idx_gal_tmp = find(~idx_probe(u,:));
            personID_gal_tmp = personID_test(idx_gal_tmp);
            upID_gal_tmp = unique(personID_gal_tmp);
            idx_gal = zeros(1,numel(upID_gal_tmp));
            for up = 1:numel(upID_gal_tmp)
                tmpid = find(personID_gal_tmp==upID_gal_tmp(up));
                idx_gal(up) = tmpid(randsample(numel(tmpid),1));
            end
            idx_gal = idx_gal_tmp(idx_gal);
            idx_gallery(u,:) = ismember(1:numel(personID_test),idx_gal);
        end
    case 'market'
        idx_train = split.trainIdx;
        idx_test = ~split.trainIdx;
        idx_probe = split.probeIdx;
        idx_gallery = ~split.probeIdx;
    case 'ilidsvid'
        tSize = 150;
        personID=unique(personID,'stable');
        personID=[personID personID];
        camID=[ones(1,300) 2*ones(1,300)];
        tmpSplit = split(s,:);
        id_train = tmpSplit(1:tSize); % first half to train
        id_train = personID(id_train);
        id_test = tmpSplit(tSize+1:end); 
        id_test = personID(id_test); 
        idx_train = ismember(personID,id_train);
        idx_test = ismember(personID,id_test);
        camID_test = camID(idx_test);
        idx_gallery = camID_test == 2;
        idx_probe = camID_test == 1;                
    case 'prid'
        tSize = 89;
        personID=unique(personID,'stable');
        personID=[personID personID];
        camID=[ones(1,178) 2*ones(1,178)];
        tmpSplit = split(s,:);
        id_train = tmpSplit(1:tSize); % first half to train
        id_train = personID(id_train);
        id_test = tmpSplit(tSize+1:end); 
        id_test = personID(id_test); 
        idx_train = ismember(personID,id_train);
        idx_test = ismember(personID,id_test);
        camID_test = camID(idx_test);
        idx_gallery = camID_test == 2;
        idx_probe = camID_test == 1;
    case 'saivt'
        if(strcmp(data.pair,'38'))
            tSize = 31;
            camID=[ones(1,99) 2*ones(1,99)];
        else
            tSize = 33;
            camID=[ones(1,103) 2*ones(1,103)];
        end
        personID=unique(personID,'stable');
        personID=[personID personID];        
        tmpSplit = split(s,:);
        id_train = tmpSplit(1:tSize); % first half to train
        id_train = personID(id_train);
        id_test = tmpSplit(tSize+1:end); 
        id_test = personID(id_test); 
        idx_train = ismember(personID,id_train);
        idx_test = ismember(personID,id_test);
        camID_test = camID(idx_test);
        idx_gallery = camID_test == 2;
        idx_probe = camID_test == 1;        
    case 'ward'
        tSize=35;
        personID=unique(personID,'stable');
        personID=[personID personID];
        camID=[ones(1,70) 2*ones(1,70)];
        tmpSplit = split(s,:);
        id_train = tmpSplit(1:tSize); % first half to train
        id_train = personID(id_train);
        id_test = tmpSplit(tSize+1:end); 
        id_test = personID(id_test); 
        idx_train = ismember(personID,id_train);
        idx_test = ismember(personID,id_test);
        camID_test = camID(idx_test);
        idx_gallery = camID_test == 2;
        idx_probe = camID_test == 1;
    case 'raid'
        tSize=21;
        if(strcmp(data.pair,'12'))
            camID=[ones(1,43) 2*ones(1,43)];
        elseif(strcmp(data.pair,'13'))
            camID=[ones(1,42) 2*ones(1,42)];
        else
            camID=[ones(1,42) 2*ones(1,42)];
        end
        personID=unique(personID,'stable');
        personID=[personID personID];        
        tmpSplit = split(s,:);
        id_train = tmpSplit(1:tSize); % first half to train
        id_train = personID(id_train);
        id_test = tmpSplit(tSize+1:end); 
        id_test = personID(id_test); 
        idx_train = ismember(personID,id_train);
        idx_test = ismember(personID,id_test);
        camID_test = camID(idx_test);
        idx_gallery = camID_test == 2;
        idx_probe = camID_test == 1;
    case 'caviar'
        tSize = 25;
        upersonID=unique(personID,'stable');
        upersonID=[upersonID upersonID];
        camID=[ones(1,50) 2*ones(1,50)];
        tmpSplit = split(s,:);
        id_train = tmpSplit(1:tSize); % first half to train
        id_train = upersonID(id_train);
        id_test = tmpSplit(tSize+1:end); 
        id_test = upersonID(id_test); % include distractors
        idx_train = ismember(upersonID,id_train);
        idx_test = ismember(upersonID,id_test);
        camID_test = camID(idx_test);
        idx_gallery = camID_test == 2;
        idx_probe = camID_test == 1;
    case 'v47'
        tSize = 23;
        upersonID=unique(personID,'stable');
        upersonID=[upersonID upersonID];
        camID=[ones(1,47) 2*ones(1,47)];
        tmpSplit = split(s,:);
        id_train = tmpSplit(1:tSize); % first half to train
        id_train = upersonID(id_train);
        id_test = tmpSplit(tSize+1:end); 
        id_test = upersonID(id_test); % include distractors
        idx_train = ismember(upersonID,id_train);
        idx_test = ismember(upersonID,id_test);
        camID_test = camID(idx_test);
        idx_gallery = camID_test == 2;
        idx_probe = camID_test == 1;
    case '3dpes'
        tSize = 96;        
        upersonID=unique(personID,'stable');                
        tmpSplit = split(s,:);
        id_train = upersonID(tmpSplit(1:tSize)); % first half to train        
        id_test = tmpSplit(tSize+1:end); 
        id_test = upersonID(id_test); 
        idx_train = ismember(personID,id_train);
        idx_test = ismember(personID,id_test);
        camID_test = camID(idx_test);
        personID_test = personID(idx_test);
        uID_test = unique(personID_test);
        idx_gallery = zeros(1,numel(personID_test));
        idx_probe = zeros(1,numel(personID_test));
        for i = 1:tSize
            tmpidx = find(personID_test==uID_test(i));
            tmpGal = tmpidx(randsample(numel(tmpidx),1));
            tmpProb = setdiff(tmpidx,tmpGal);
            idx_gallery(tmpGal) = 1;
            idx_probe(tmpProb) = 1;
        end
        idx_gallery = logical(idx_gallery);
        idx_probe = logical(idx_probe);
end