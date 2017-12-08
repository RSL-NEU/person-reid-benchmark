function score = nAuc(rank,maxr)
% normalized AUC(area under the curve) 
maxr = min(maxr,numel(rank));
nf = rank(end);
if rank(end)~=1 && rank(end)~=100
    warning('Unproper cmc ranking score!');
end
score = sum(rank(1:maxr))./(nf*maxr);