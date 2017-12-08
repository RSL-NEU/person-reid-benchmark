function [p] = evalensemble(X,ensemble,p)
	% extract from ensemble
	learningrates = ensemble{1};
	niters = length(learningrates);
	treesperiter = length(ensemble) - 1;
    label_length = size(ensemble{2}{1},2) - 3;
    useMultilabelTree = label_length > 1;
    
	% initialize predictions
	n = length(X(:,1));
    if nargin < 3
        if (useMultilabelTree) 
            p = zeros(n, label_length);
        else
            p = zeros(n, treesperiter);
        end
    end


    % compute predictions from trees
    for i=1:niters,
        learningrate = learningrates(i);
        if (useMultilabelTree)
            p = p + learningrate * evaltree(X, ensemble{2}{i});
        else
            for t=1:treesperiter
                p(:,t) = p(:,t) + learningrate * evaltree(X, ensemble{t+1}{i});
            end
        end
    end
    
end
