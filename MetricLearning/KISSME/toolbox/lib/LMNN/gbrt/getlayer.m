function [layer,rowindices,maxdepth] = getlayer(tree,depth)
	maxdepth = log2(size(tree,1)+1);
	rowindices = 2^(depth-1):2^depth-1;
	layer = tree(rowindices,:);
end
