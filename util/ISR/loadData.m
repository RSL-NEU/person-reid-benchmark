%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) Copyright 2012-13  MICC - Media Integration and Communication Center,
% University of Florence. Giuseppe Lisanti <giuseppe.lisanti@unifi.it> and 
% Iacopo Masi <iacopo.masi@unifi.it>. Fore more details see URL 
% http://www.micc.unifi.it/lisanti/source-code/re-id
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


switch datasetname
   case 'VIPeR'
      maxRankDisplay = 316;
      load('data/VIPeR_trials.mat')
      load('data/VIPeR_features.mat');
      idPersonsVip = unique(mapping);
   case 'CAVIARa'
      load('data/CAVIARa_features.mat');
      try
         load(['data/CAVIARa_trials_G' num2str(maxNumTemplate) '_T' num2str(maxNumTemplateTest) '.mat']);
      catch err
         disp('Combination not available');
      end
      maxRankDisplay = 30;
    otherwise
        disp('Dataset not found: currently this dataset is not yet supported.');
end