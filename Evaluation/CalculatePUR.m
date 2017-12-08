% Calculate the "proportion of uncertainty removed" as in cvpr 2013 paper
% "Local Fisher Discriminant Analysis for Pedestrian Re-Identification"
% By Fei Xiong, 
%    ECE Dept, 
%    Northeastern University 
%    2013-11-04
% r: cumulated matching propability, which represents the probability that 
%    the choice of the first r-th rank choice is correct.
% num_ref: the total number of reference ID.
% Note that the equaion 16 in the paper have a sign typo.
function PUR = CalculatePUR(r,num_ref)
r =[r(1) r(2:end) - r(1:end-1)]; % compute the pdf
% equation 16, the minus operation in that equation should be plus
PUR =(log2(num_ref) + sum(r.*log2(r+1e-8)))/log2(num_ref); 
return;
