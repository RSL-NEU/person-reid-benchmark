function [p] = evaltree(X,tree)

	% start at root node
	n = ones(size(X(:,1)));
	
	% update nodes by descending tree
	[~,~,maxdepth] = getlayer(tree,1);
	for d=1:maxdepth-1,
		layer = getlayer(tree,d);
		bestfeatures = layer(:,1);
		Fv = bestfeatures(n);
		bestsplits = layer(:,2);
		Sv = bestsplits(n);
		Iv = sub2ind(size(X),1:size(X,1),Fv');
		Vv = X(Iv);
%         		nNotInLeaf = tree(n,3) < 10^30; % replaced by the following line --Gao
        nNotInLeaf = tree(2^(d-1)-1+n,3) < 10^30; % a change here
        n(nNotInLeaf) = 2*n(nNotInLeaf) - (Vv(nNotInLeaf)' < Sv(nNotInLeaf));
			% Fv(n) = Feature Index
			% Vv(n) = Feature Value
            % Sv(n) = Split Value
	end
    
	% determine predictions from leaf labels
	leafnodes = getlayer(tree,maxdepth); %Stores the last Layer of tree into leafnodes
    outputlabels = leafnodes(:,4:end); %Stores the predictions of the leafnodes into outputlabels
    p = outputlabels(n,:); %Stores the prediction for each instance into p
end