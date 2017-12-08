% Get features according to the split

function [galleryFeatures,probeFeatures,galleryId,probeId]=getFeaturesSplits(mode,gf,pf,gid,pid,dataset,s)

galleryFeatures=gf;
probeFeatures=pf;
galleryId=gid;
probeId=pid;

person_ids=unique(gid);person_ids1=unique(pid);
assert(isequal(person_ids,person_ids1));


load(strcat('Split_',dataset.name,dataset.pair));

switch dataset.name
    case 'viper'
        if(strcmp(mode,'train'))
            reqIds=split(s,1:316);
        else
            reqIds=split(s,317:632);
        end
    case 'grid'
        if(strcmp(mode,'train'))
            reqIds=split(s,1:125);
        else
            reqIds=split(s,126:250);
        end
    case 'ilids'
        if(strcmp(mode,'train'))
            reqIds=split(s,1:150);
        else
            reqIds=split(s,151:300);
        end
    case 'prid'
        if(strcmp(mode,'train'))
            reqIds=split(s,1:89);
        else
            reqIds=split(s,90:178);
        end
    case 'saivt'
        switch dataset.pair
            case '38'
                if(strcmp(mode,'train'))
                    reqIds=split(s,1:31);
                else
                    reqIds=split(s,32:99);
                end
            case '58'
                if(strcmp(mode,'train'))
                    reqIds=split(s,1:33);
                else
                    reqIds=split(s,34:103);
                end
        end
end

galleryFeatures1=[];
galleryId1=[];
probeFeatures1=[];
probeId1=[];

for i=1:length(reqIds)
    currId=person_ids(reqIds(i));
    ids=find(galleryId==currId);
    galleryFeatures1=[galleryFeatures1;galleryFeatures(ids,:)];
    galleryId1=[galleryId1 i*ones(1,length(ids))];
    ids=find(probeId==currId);
    probeFeatures1=[probeFeatures1;probeFeatures(ids,:)];
    probeId1=[probeId1 i*ones(1,length(ids))];
end

galleryFeatures=galleryFeatures1;
galleryId=galleryId1;
probeFeatures=probeFeatures1;
probeId=probeId1;


