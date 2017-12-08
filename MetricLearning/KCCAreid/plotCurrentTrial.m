%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) Copyright 2013-14  MICC - Media Integration and Communication Center,
% University of Florence. Giuseppe Lisanti <giuseppe.lisanti@unifi.it> and
% Iacopo Masi <iacopo.masi@unifi.it>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

color = colors( mod(nt,length(colors)) );
figure(100);hold on;plotCMCcurve(cmcCurrent,color,'.',...
   [datasetname ' N=' num2str(maxNumTemplate)]);
drawnow