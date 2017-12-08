function displayname = datasetMap(datasets)

if ~iscell(datasets)
    datasets = {datasets};
end
for d = 1:numel(datasets)    
    switch datasets{d}
        case '3dpes'
            tmpname='3DPeS';
        case 'airport'
            tmpname='Airport';
        case 'cuhk_detected'
            tmpname='CUHK03';
        case 'cuhk01'
            tmpname='CUHK01';
        case 'cuhk02'
            tmpname='CUHK02';
        case 'grid'
            tmpname='GRID';
        case 'ilidsvid'
            tmpname='iLIDSVID';
        case 'market'
            tmpname='Market1501';
        case 'prid'
            tmpname='PRID';
        case 'raid_12'
            tmpname='RAiD-12';
        case 'raid_13'
            tmpname='RAiD-13';
        case 'raid_14'
            tmpname='RAiD-14';
        case 'saivt_38'
            tmpname='SAIVT-38';
        case 'saivt_58'
            tmpname='SAIVT-58';
        case 'viper'
            tmpname='VIPeR';
        case 'ward_12'
            tmpname='WARD-12';
        case 'ward_13'
            tmpname='WARD-13';
        case 'caviar'
            tmpname='CAVIAR';
        case 'hda'
            tmpname='HDA+';
        case 'v47'
            tmpname='V47';
        case 'DukeReIDmix'
            tmpname = 'DukeMTMC4ReID';
        case 'DukeReIDpair'
            tmpname = 'DukeMTMC4ReID';           
    end
    displayname{d} = tmpname;
end
