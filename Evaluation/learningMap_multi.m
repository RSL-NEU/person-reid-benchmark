function [displearn,dispmatch] = learningMap_multi(algoname)

if ~iscell(algoname)
    algoname = {algoname};
end
for a = 1:numel(algoname) 
    % change name
    tmpname = strrep(algoname{a},'klfda','kLFDA');
    tmpname = strrep(tmpname,'kmfa','kMFA');
    tmpname = strrep(tmpname,'kpcca','kPCCA');
    tmpname = strrep(tmpname,'rpcca','rPCCA');
    tmpname = strrep(tmpname,'xqda','XQDA');
    tmpname = strrep(tmpname,'mfa','MFA');
    tmpname = strrep(tmpname,'lfda','LFDA');
    tmpname = strrep(tmpname,'fda','FDA');    
    tmpname = strrep(tmpname,'svmml','SVMML');
    tmpname = strrep(tmpname,'pcca','PCCA');
    tmpname = strrep(tmpname,'kissme','KISSME');
    tmpname = strrep(tmpname,'prdc','PRDC');
    tmpname = strrep(tmpname,'ranksvm','RankSVM');
    tmpname = strrep(tmpname,'itml','ITML');
    tmpname = strrep(tmpname,'lmnn','LMNN');
    tmpname = strrep(tmpname,'kcca','kCCA');
    tmpname = strrep(tmpname,'l2','L2');
    
    if any(strfind(tmpname,'featureAverage'))
        tmpname = strrep(tmpname,'featureAverage','aver');
        tmpname = [tmpname(6:end),'_aver'];
    end
    if any(strfind(tmpname,'clustering'))
        tmpname = strrep(tmpname,'clustering','clust');
        tmpname = tmpname(7:end);
    end
    if any(strfind(tmpname,'linear')) || any(strfind(tmpname,'chi2')) || ...
            any(strfind(tmpname,'chi2-rbf')) || any(strfind(tmpname,'exp'))
        tmp = strfind(tmpname,'_');
        tmpname = strrep(tmpname,'_','-');
        tmpname = [tmpname(1:tmp(1)-1) '$_{' tmpname(tmp(1)+1:tmp(2)-1) '}$' ...
                    tmpname(tmp(2):end)];
        tmpname = strrep(tmpname,'linear','\ell');
        tmpname = strrep(tmpname,'chi2-rbf','R_{\chi^2}');
        tmpname = strrep(tmpname,'chi2','\chi^2');
    else
        tmpname = strrep(tmpname,'_','-');
    end
    tmp = strfind(tmpname,'-');
    
    tmpname = strrep(tmpname,'aver','AVER');
    tmpname = strrep(tmpname,'rnp','RNP');
    tmpname = strrep(tmpname,'ahisd','AHISD');
    tmpname = strrep(tmpname,'sanp','SANP');
    tmpname = strrep(tmpname,'srid','SRID');
    tmpname = strrep(tmpname,'isr','ISR');
    
    dispmatch{a} = tmpname(tmp+1:end);
    displearn{a} = tmpname(1:tmp-1);
end