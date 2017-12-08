
data='cuhk02';

if(strcmp(data,'saivt')||strcmp(data,'ward')||strcmp(data,'raid'))
    pair='14'; % SAIVT: 38, 58, WARD: 12, 13, RAiD: 12, 13, 14
end

nSplits=10;

switch data
    case 'viper'
        nPersons=632;
    case 'grid'
        nPersons=250;
    case 'ilids'
        nPersons=300;
    case 'prid'
        nPersons=178;
    case 'caviar'
        nPersons=50;
    case 'raid'
        switch pair
            case '12'
                nPersons=43;
            case {'13','14'}
                nPersons=42;            
        end
    case 'ward'
        nPersons=70;
    case 'saivt'
        switch pair
            case '38'
                nPersons=99;
            case '58'
                nPersons=103;
        end
    case '3dpes'
        nPersons = 192;
    case 'v47'
        nPersons=47;
    case 'cuhk01'
        nPersons = 971;
    case 'cuhk02'
        nPersons = 1816;
end

if(strcmp(data,'saivt'))
    switch pair
        case '38'
            nTraining=31;
        case '58'
            nTraining=33;
    end
elseif(strcmp(data,'raid'))
    nTraining=21;
elseif(strcmp(data,'v47'))
    nTraining=23;
elseif(strcmp(data,'cuhk01'))
    nTraining = 485;
elseif(strcmp(data,'cuhk02'))
    nTraining = 908;
else
    nTraining=nPersons/2;
end

for i=1:nSplits
    train=randperm(nPersons,nTraining);
    test=setdiff(1:nPersons,train);
    split(i,:)=[train test];
end

if(strcmp(data,'saivt')||strcmp(data,'ward')||strcmp(data,'raid'))
    save(strcat('Split_',data,'_',pair,'.mat'),'split');
else
    save(strcat('Split_',data,'.mat'),'split');
end

