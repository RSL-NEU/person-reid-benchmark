%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) Copyright 2012-13  MICC - Media Integration and Communication Center,
% University of Florence. Giuseppe Lisanti <giuseppe.lisanti@unifi.it> and 
% Iacopo Masi <iacopo.masi@unifi.it>. Fore more details see URL 
% http://www.micc.unifi.it/lisanti/source-code/re-id
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Force figure in the dock
set(0,'DefaultFigureWindowStyle','docked');

switch datasetname
   case 'VIPeR'
      open('figure/CMCviper.fig');
   case 'CAVIARa'
      if maxNumTemplate ~= 1
         open('figure/CMCcaviar_mvm.fig');
      else
         open('figure/CMCcaviar_svs.fig');
      end
end