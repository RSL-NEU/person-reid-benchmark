function [args] = preprocesslayer_sqrimpurity(data, options) 
	% confirm necessary data or options
	assert(isfield(data,'numfeatures'))
	assert(isfield(data,'n'));
	assert(isfield(data,'y'));
	assert(isfield(data,'depth'));
	assert(isfield(data,'tree'))
	
	% compute counts for each node
	numnodes = 2^(data.depth-1);
    
    % Allocating space for l_infty.
    m_infty = -99999 * ones(1, numnodes);
    l_infty = -99999 * ones(numnodes, size(data.y,2));
	for i=1:numnodes
		m_infty(i) = sum(data.n==i);
		l_infty(i,:) = sum(data.y(data.n==i,:),1);
    end
	
	l_infty = l_infty';
	
	

    
	% get parents
	parents = getlayer(data.tree,data.depth);
	
	% include feature costs
	if isfield(options,'featurecosts'),
		featurecosts = options.featurecosts;
    else
		featurecosts = zeros(data.numfeatures,1);
	end
	
	
	% create impurity evalution argument
	args = {m_infty, l_infty, parents(:,4:end)', featurecosts};
end
