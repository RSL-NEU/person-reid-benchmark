%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) Copyright 2012-13  MICC - Media Integration and Communication Center,
% University of Florence. Giuseppe Lisanti <giuseppe.lisanti@unifi.it> and 
% Iacopo Masi <iacopo.masi@unifi.it>. Fore more details see URL 
% http://www.micc.unifi.it/lisanti/source-code/re-id
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all force
set(0,'DefaultFigureWindowStyle','docked');

%% Dataset Parameters
datasetname = 'CAVIARa'; % Dataset name: CAVIARa or VIPeR

%% Here some supported configurations
% 1) Viper  SvsS only maxNumTemplate==maxNumTemplateTest=1
% 2) CAVIARa SvsS and MvsM (N=5) so maxNumTemplate==maxNumTemplateTest=5
maxNumTemplate = 5;     % Number of exemplars in the GALLERY
maxNumTemplateTest = 5; % Number of exemplars in the PROBE
% You can change the number of splits (trials) below, at row 49 and 54.

%% Test SPAMS installation.
addpath('spams-matlab/build');
if ~testSPAMS()
    disp(['SPAMS test failed. You must have a working installation of SPAMS version 2.3']);
    disp(['Follow the instructions in the spams-matlab/ directory to compile SPAMS.']);
    disp(['See http://spams-devel.gforge.inria.fr/ for more info on SPAMS']);
    return
end

%% Visualization Parameters
waitBarON = 1;
colors = repmat('rgbkmcyrgbk',1,200);
markers = repmat('+o*.xsd^v<>ph',1,200);

%% Algorithm parameters
param.pos    = true; % positive constraints in the reconstruction
param.mode   = 2;    % please see 'help mexLasso' in the cmd line
param.lambda = 0.2;  % algorithm param as reported at the end of Section 4.1
nIter = Inf; % number of iteration in the algorithm; Inf means till the end.
             % if you set 10, means up to rank-10.

% Loading the pre-computed data
disp('Loading data...');
loadData

% Preparing structure for CMC curve
if strcmp('VIPeR',datasetname)
   nTrial = 10;
   numClasses = 316;
   cmc = zeros(316,3);
   cmc(:,1) = 1:316;
else
   nTrial = 50;
   numClasses = length(idPersons);
   cmc = zeros(length(idPersons),3);
   cmc(:,1) = 1:length(idPersons);
end

% Initialize structure for CMC computation
errorSortArr = [];
labelSortArr = [];

%% Loop on trial (splits)
for nt=1:nTrial
   disp(['Trial ' num2str(nt) ' of ' num2str(nTrial) ' ...']);
   
   % Load trial configuration if availble
   if strcmp('VIPeR',datasetname)
      idPersons=[];
      randID = randIDarr{nt};
      for r=1:length(randID)
         idxr = find(randID(r) == idPersonsVip);
         idPersons = [ idPersons idPersonsVip(idxr)];
      end
      cmcCurrent = zeros(316,3);
      cmcCurrent(:,1) = 1:316;
   else
      cmcCurrent = zeros(length(idPersons),3);
      cmcCurrent(:,1) = 1:length(idPersons);
   end
      
   % Build the super base (in the paper it refers to the matrix T eq.5 )
   buildSuperBase_demo
   
    % Normalizing the super base such that each column l2-norm is one.
   disp('Normalizing super-base...');
   [Dnorm norms] = normalizeBase(D);
   
   % The matrix of test features are already loaded with the mat file.
   testidAll = [];
   testLabel = [];
   for tt=1:length(idPersons)
      idsTest = testId(tt).ids;
      testidAll = [testidAll idsTest'];
      testLabel = [testLabel idPersons(tt)*ones( 1,length(idsTest) ) ];
   end
   
   % Normalize the feature matrix.
   featuresTest = featuresAll(testidAll,:);
   featuresTest = featuresTest';
   [featuresTest featuresTestNorms] = normalizeBase(featuresTest);
   
   % First iteration. It solves eq.6 (for all the probes jointly in order
   %to save time). So now we have a matrix of coefficients. Each column 
   % is a coefficient vector...
   Alphas = full( mexLasso(single(featuresTest),single(Dnorm),param) );
   
   % ...now for each person in the dataset perform re-weighted iterative 
   % ranking
   disp('Testing...');
   if waitBarON
      hwait = waitbar(0,'Testing...');
   end
   
   counts=1;
   
   %% SvsS modality
   if maxNumTemplate == maxNumTemplateTest && maxNumTemplate == 1
      % For each person in the test set
      for eachA=1:size(featuresTest,2)
         
         % Select the feature vector
         eachfeaturesTest = featuresTest(:,eachA);
         % Select the reconstructed coefficients
         Alpha = Alphas(:,eachA);
         % Check if reconstruction worked
         checkSparsityError
         %% Perform iterative re-weighted ranking
         [errorSort finalLabel] = ...
         sparseClassifyIterateWeighted_demo(eachfeaturesTest,Dnorm,Alpha,...
                                            param,idPersons,labels,nIter,...
                                            maxNumTemplate);
         %% Evaluation
         [cmc cmcCurrent] = evaluateCMC_demo(testLabel(eachA),finalLabel...
                                             ,cmc,cmcCurrent); 
         if waitBarON
            waitbar(eachA/size(featuresTest,2),hwait);
         end
      end
   else
   %% MvsM modality
   uniqueLabel=unique(testLabel);
   for eachA=1:size(uniqueLabel,2)
      
      % Select the index of the instances of that person
      idxSamePerson = find( uniqueLabel(eachA) == testLabel );
      % Select the feature vector of that group
      eachfeaturesTest = featuresTest(:,idxSamePerson);
      % Select the reconstructed coefficients
      Alpha = Alphas(:,idxSamePerson);
      % Check if reconstruction worked
      checkSparsityError
      
      %% Perform iterative re-weighted ranking
      [errorSort finalLabel] = ...
      sparseClassifyIterateWeighted_demo(eachfeaturesTest,Dnorm,Alpha,...
                                         param,idPersons,labels,nIter,...
                                         maxNumTemplate);
      %% Evaluation
      [cmc cmcCurrent] = evaluateCMC_demo(uniqueLabel(eachA),finalLabel...
                                          ,cmc,cmcCurrent);
      if waitBarON
         waitbar(eachA/size(featuresTest,2),hwait);
      end
   end
   end
   
   if waitBarON
      close(hwait); 
   end
   
   %% Plot the result of each trial (split) for visualization purposes.
   plotCurrentTrial
   clear D;
   
end

%% Open the right figure for the dataset
openRightFigure

%% Plotting the final CMC curve
cmcranked = cmc(1:maxRankDisplay,:);
gcf;hold on;plotCMCcurve(cmcranked,'b','.',...
   [datasetname ' N=' num2str(maxNumTemplate)]);
% Getting the normalized Area under the Curve (nAUC)
nAUC = getnAUC(cmc);
hold on;
xnauc = maxRankDisplay-20;
ynauc = cmc(1,2)/cmc(1,3)*100.;
text(xnauc,ynauc,['nAUC ' num2str(nAUC)],'FontSize',20,'FontName','Times');
% Tuning the legend
legend('-DynamicLegend');
legend('Location','SouthEast');