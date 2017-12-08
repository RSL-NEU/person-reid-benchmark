%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) Copyright 2012-13  MICC - Media Integration and Communication Center,
% University of Florence. Giuseppe Lisanti <giuseppe.lisanti@unifi.it> and 
% Iacopo Masi <iacopo.masi@unifi.it>. Fore more details see URL 
% http://www.micc.unifi.it/lisanti/source-code/re-id
%
%
%[errorSort idxSort] = sparseClassify(eachfeaturesTest,Dnorm,...
%    Alpha,idPersons,labels)
%
% The function computes the error and the order derived from the eq.7 of
% our paper of Sparse Discriminative Classifier.
%
% Input
%
% eachfeaturesTest: the current feature test to rank
% Dnorm: the normalized superbase (gallery set)
% Alpha: the current reconstructed coefficient from which to start.
% param: the param reported for the optimization. please see 'help
% mexLassoWeighted' for that.
% labels: the label for each person in Dnorm
% idPersons: the id for each person
%
% Output
%
% errorSort: the error used to sort PART of the gallery (note that is
% a monotone function). The error equation is that of eq.8.
%
% idxSort: the indexes of sorting.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [errorSort idxSort] = sparseClassify(eachfeaturesTestAll,Dnorm,...
   Alphas,idPersons,labels)

errors=[];
for t=1:size(eachfeaturesTestAll,2)
   Alpha=Alphas(:,t);
   eachfeaturesTest = eachfeaturesTestAll(:,t);
   % Check if the Alpha are all zeros. If so, break!
   checkSparsityError
   % Repeat the Alpha of that person for each other person
   Alphaband = repmat(Alpha,1,length(idPersons));
   
   % Band the Alpha of that person for each other persons
   for p = 1:length(idPersons)
      I = find(idPersons(p) ~= labels & labels ~=0); %% label == 0 are
      %the trivial templates, if used
      Alphaband(I,p) = 0; %#ok<FNDSB>
   end
   
   % Compute column-wise residuals
   residual = computeResidualColumnWise(eachfeaturesTest,idPersons,Dnorm,Alphaband);
   % Compute column-wise error with l2-norm...
   error = normColumnWise(residual,2);
   errors = [errors error];
end
% ...and pick the band with minimum error.
finalerror = min(errors,[],2);
[errorSort idxSort]= sort(finalerror,'ascend');
return