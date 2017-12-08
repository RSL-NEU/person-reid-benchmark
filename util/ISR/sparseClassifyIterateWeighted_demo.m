%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) Copyright 2012-13  MICC - Media Integration and Communication Center,
% University of Florence. Giuseppe Lisanti <giuseppe.lisanti@unifi.it> and 
% Iacopo Masi <iacopo.masi@unifi.it>. Fore more details see URL 
% http://www.micc.unifi.it/lisanti/source-code/re-id
%
%
% [errorSortFinal finalLabel] = sparseClassifyIterateWeighted_demo(...
%   eachfeaturesTest,Dnorm,Alpha,param,...
%   idPersons,labels,nIter,maxNumTemplate)
%
% The function gets the feature of a target to test, the normalized
% superbase Dnorm and the coefficient reconstructed in the first step. It
% also takes the params for optimization,the labels associated with the
% superbase and the max number of exemplar per person in the gallery.
% It returns the final labels vector as ranked by our algorithm along with
% the final error used to sort the labels.
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
% nIter: number of iteration to perform. If it's set to 'Inf' it will run
% till all the gallery is not ranked. Otherwise we can bound the # of
% iteration up to a given rank.
% maxNumTemplate: the number of exemplars per person in the gallery.
%
% Output
%
% errorSortFinal: the final error used to sort the gallery (note that is
% not a monotone function). The error equation is that of eq.8.
%
% finalLabel: a vector that contains ordered labels reported from our
% method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [errorSortFinal, finalLabel, num_iter] = sparseClassifyIterateWeighted_demo(...
   eachfeaturesTest,Dnorm,Alpha,param,...
   idPersons,labels,nIter,maxNumTemplate)

% Initializing structures
errorSortFinal = [];
finalLabel = [];
count = 0;
w3 = zeros(size(Alpha,1),size(Alpha,2));
max_inf = 1;
num_iter = [];
%% Iterate till the superbase has something and nIter has not reached the end
while length(unique(finalLabel)) ~= length(idPersons)
   
   count = count + 1;
   
   % Getting the eps of eq.10
   % The $\varepsilon$ is chosen to be slightly smaller than the expected
   % nonzero magnitudes of Alpha.
   epsilon = min(Alpha(find(Alpha~=0)));
   
   % Order the person that are already ranked.
   if(isempty(epsilon))
      if(length(unique(finalLabel)) ~= length(idPersons))
         ind_to_insert = setdiff(unique(idPersons),unique(finalLabel));
         if ~isempty(max(errorSortFinal))
            err_to_insert = ones(length(ind_to_insert),1).*max(errorSortFinal);
         else
            err_to_insert = ones(length(ind_to_insert),1).*1;
         end
         finalLabel = [finalLabel ind_to_insert];
         errorSortFinal = [errorSortFinal; err_to_insert];
         num_iter = [num_iter count.*ones(1,length(err_to_insert))];
      end
      break;
   end
   
   %% Soft weighting for robust ranking (section 5.1)
   % Computing the weights for this iteration, eq.11.
   w = 1./(Alpha+epsilon); % eq.10
   w = w./max(w(:));
   
   % solving eq.11 with soft weighting
   Alpha = full(mexLassoWeighted(single(eachfeaturesTest),single(Dnorm),single(w),param));
   
   %% Perform sparse Classification
   [errorSort idxSort] = sparseClassify(eachfeaturesTest,Dnorm,Alpha,idPersons,labels);
   
   % Find the index to retain and that one to remove
   % (Find non-zero coefficients )
   maxe = max(errorSort);
   idxnull = find(errorSort == maxe);
   idxtotake = 1:idxnull(1)-1;
   
   %% Compose final error and finalLabel
   errorSortFinal = [errorSortFinal; errorSort(idxtotake)];
   finalLabel = [finalLabel idPersons(idxSort(idxtotake))];
   num_iter = [num_iter count.*ones(1,length(idxtotake))];
   
   if(isempty(Alpha(find(Alpha~=0))))
      if(length(unique(finalLabel)) ~= length(idPersons))
         ind_to_insert = setdiff(unique(idPersons),unique(finalLabel));
         err_to_insert = ones(length(ind_to_insert),1).*max(errorSortFinal);
         finalLabel = [finalLabel ind_to_insert];
         errorSortFinal = [errorSortFinal; err_to_insert];
         num_iter = [num_iter count.*ones(1,length(err_to_insert))];
      end
      break;
   end
   
   %% Hard re-weighting for ranking completeness (section 5.2)
   % Computing the weights for this iteration, eq.11 and eq.12.
   w2 = zeros(size(Alpha,1),size(Alpha,2));
   w2(find(Alpha>0)) = max_inf;
   w2(find(Alpha==0)) = 1/size(Alpha,1);
   w3=w3+w2;
   
   % solving eq.11 with hard weighting
   Alpha = full(mexLassoWeighted(single(eachfeaturesTest),single(Dnorm),single(w3),param));
   
   tmp = histc(finalLabel,1:max(idPersons));
   ind_tmp = find( tmp > maxNumTemplate );
   if(~isempty(ind_tmp))
      break;
   end
   
end

%% Compose final error and finalLabel
tmp = histc(finalLabel,1:max(idPersons));
ind_tmp = find( tmp > 1 );
if(~isempty(ind_tmp))
   for i_d=1:length(ind_tmp)
      ind_rem = find(finalLabel==ind_tmp(i_d));
      ind_rem(1)=[];
      finalLabel(ind_rem)=[];
      errorSortFinal(ind_rem)=[];
   end
   if(length(unique(finalLabel)) ~= length(idPersons))
      ind_to_insert = setdiff(unique(idPersons),unique(finalLabel));
      err_to_insert = ones(length(ind_to_insert),1).*max(errorSortFinal);
      finalLabel = [finalLabel ind_to_insert];
      errorSortFinal = [errorSortFinal; err_to_insert];
      num_iter = [num_iter count.*ones(1,length(err_to_insert))];
   end
end

%% Check if the number of iteration are not completed: leave the order
%% as is if not.
if count >= nIter
   errorSortFinal = [errorSortFinal; errorSort(idxnull)];
   finalLabel = [finalLabel idPersons_back(idxSort(idxnull))];
   num_iter = [num_iter count.*ones(1,length(idxnull))];
end

%% Final Check on the label and finalLabel
if length(unique(finalLabel)) ~= length(finalLabel)
   disp('warning: finalLabel not consistent!');
end

return