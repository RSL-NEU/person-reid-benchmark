%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) Copyright 2012-13  MICC - Media Integration and Communication Center,
% University of Florence. Giuseppe Lisanti <giuseppe.lisanti@unifi.it> and 
% Iacopo Masi <iacopo.masi@unifi.it>. Fore more details see URL 
% http://www.micc.unifi.it/lisanti/source-code/re-id
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Building the super-Base
count=0;
if ~exist('D','var')
    % Initializing
    D=[]; labels = []; trainidAll=[];
    
    % get the split configuration
    trainId = trialStat(nt).trainId;
    testId = trialStat(nt).testId;
    
    disp('Building the super-basis...');
    if waitBarON
        hwait = waitbar(0,'Building the super-basis...');
    end
    countTt = 0;
    
    % loop over the perons
    for tt=1:length(idPersons)
        
        countTt = countTt + 1;
        
        trainIdCurrent = trainId(countTt).ids;
        
        % Compose the base using the precomputed features
        features = featuresAll(trainIdCurrent,:);
        D=[D features']; % in D we store the super-base.
        trainidAll = [trainidAll trainId(countTt).ids'];
        labels = [labels idPersons(tt)*ones(1,size(features' ,2))]; %here
        %we store the label, for each column of D.
        
        if waitBarON
            waitbar(tt/length(idPersons),hwait)
        end
    end
    
    if waitBarON
        close(hwait)
    end
else
    disp('Super base already loaded.');
end

