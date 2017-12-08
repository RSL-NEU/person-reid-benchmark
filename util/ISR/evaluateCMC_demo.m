%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) Copyright 2012-13  MICC - Media Integration and Communication Center,
% University of Florence. Giuseppe Lisanti <giuseppe.lisanti@unifi.it> and 
% Iacopo Masi <iacopo.masi@unifi.it>. Fore more details see URL 
% http://www.micc.unifi.it/lisanti/source-code/re-id
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [cmc cmcCurrent] = evaluateCMC_demo(gtLabel,finalLabel...
                                             ,cmc,cmcCurrent)
%% Evaluation of Cumulative Matching Curves (CMC)
for l=1:length(finalLabel)
   tp = any( gtLabel == finalLabel(1:l) );
   cmc(l,2) = cmc(l,2) + tp;
   cmc(l,3) = cmc(l,3) + 1;
   cmcCurrent(l,2) = cmcCurrent(l,2) + tp;
   cmcCurrent(l,3) = cmcCurrent(l,3) + 1;
end
