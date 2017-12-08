function dispalgo = learningMap(algoname)

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
    tmpname = strrep(tmpname,'kcca','kCCA');
    
    tmpname = strrep(tmpname,'l2','L2');
    
    % change kernel
    tmpname = strrep(tmpname,'linear','$_{\ell}$');
    tmpname = strrep(tmpname,'chi2-rbf','$_{R_{\chi^2}}$');
    tmpname = strrep(tmpname,'chi2','$_{\chi^2}$');
    tmpname = strrep(tmpname,'exp','$_{exp}$');
   
    tmpname = strrep(tmpname,'_$','$');
    dispalgo{a} = tmpname;
end
