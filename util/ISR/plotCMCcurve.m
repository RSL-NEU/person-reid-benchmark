%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) Copyright 2012-13  MICC - Media Integration and Communication Center,
% University of Florence. Giuseppe Lisanti <giuseppe.lisanti@unifi.it> and 
% Iacopo Masi <iacopo.masi@unifi.it>. Fore more details see URL 
% http://www.micc.unifi.it/lisanti/source-code/re-id
%
%
% plotCMCcurve(cmc,color,marker,stringTitle)
% 
% The function plots the CMC curve associated wit the result.
%
% Input
% 
% cmc: the cmc curve
% color: the color of the curve
% stringTitle: the title for this figure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plotCMCcurve(cmc,color,marker,stringTitle)
set(gca,'FontSize',20,'FontName','Times');
recorateY = cmc(:,2)./cmc(:,3);
rankscoreX = cmc(:,1);

plot(rankscoreX, recorateY.*100. ,[color ' ' marker '-'], ...
   'Linewidth',4, 'MarkerSize', 7);% 'MarkerFaceColor',color);

title(['CMC - ' stringTitle],'FontSize',20,...
   'FontName','Times');

% Create xlabel
xlabel('Rank Score','FontSize',20,'FontName','Times');

% Create ylabel
ylabel('Recognition Rate','FontSize',20,'FontName','Times');
grid on;
end