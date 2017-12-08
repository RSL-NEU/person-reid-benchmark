%% Greedily builds a tree layer-wise

function [tree,p] = buildtree(X, Xs, Xi, y, depth, options)
    
    % Handling the multilabel case where gradient is vectorized.
    if (size(X,1) < size(y,1))
        if (mod(size(y,1),size(X,1)))
            error('Gradient elements count must divide with number of instances');
        else
            y = reshape(y,size(X,1), size(y,1) / size(X,1));
        end
    end

    % checks for required inputs.
    if nargin ~= 6,
		error('buildtree requires 6 arguments: X, Xs, Xi, y, depth, options');
    end

	% verify agreement among X, Xs, Xi, and g
	% TODO
    %Checks sizes of Xs, Xi, and y
    assert(all(size(X)==size(Xs)) & all(size(X)==size(Xi)), ...
	   'Dimensions of X, Xs, and Xi do not match');
    assert(all(size(y,1)==size(X,1)), ...
   	   'Dimensions of X and y do not match');
	[RowsX,n] = size(X);
    [RowsXs,col] = size(Xs);
    
    % Cheks that the row dimensions of X
    if RowsX ~= RowsXs,
        error('Row dimentions in X do not match row dimensions in Xs, Xi, or y')
    end
       
	% verify depth
	% TODO
	
	% get number of features
	numfeatures = size(X,2);
	
	% initialize with each instance at the root node
	n = ones(size(y,1),1);
	
	% initialize tree and compute default label
	if isfield(options,'defaultlabel'),
		defaultlabel = options.defaultlabel(y);
    else
		defaultlabel = mean(y,1);
	end
	tree = [zeros(1,3), defaultlabel]; % initialize the root node with default prediction
	
	% select function to build layer
	if isfield(options,'buildlayer'),
		buildlayer = options.buildlayer;
    else
        		buildlayer = @buildlayer_sqrimpurity_openmp;
        % 		buildlayer = @buildlayer_sqrimpurity_multif;
		% buildlayer = @buildlayer_sqrimpurity;
	end
	
	% select function to preprocess layer construction
	if isfield(options,'preprocesslayer'),
		preprocesslayer = options.preprocesslayer;
    else
		preprocesslayer = @preprocesslayer_sqrimpurity;
	end
	
	% build tree layer-wise
    for d=1:depth-1,
		% initialize stuff
		splits = [];
		impurity = [];
		labels = [];
		
		% get parent nodes
		parents = getlayer(tree,d);
		
		% prepare data for preprocessing
		% data.X = X;
		data.numfeatures = size(X,2);
		data.y = y;
		data.n = n;
		data.depth = d;
		data.tree = tree;
		
		% compute preprocessed arguments
		args = preprocesslayer(data, options);
		
		% iterate over features
        % for f=1:numfeatures,
			% [splits(f,:),impurity(f,:),labels(f,:)] = buildlayer(Xs(:,f), Xi(:,f), y, n, f, args);
        % end
		f=1:numfeatures;
        
        % if (size(tree,2) > 4)
        %     keyboard;
        % end
		[splits,impurity,labels] = buildlayer(Xs, Xi, y', n, f(:),args);
% 		[splits,impurity,labels] = buildlayer(Xs, Xi, y', n,f(:));
        
		% pick best splits for each node
		[bestimpurity,bestfeatures] = min(impurity);
        indices = sub2ind(size(impurity),bestfeatures,1:size(impurity,2));
		bestsplits = splits(indices);

		% record splits for parent nodes
		[~,pi] = getlayer(tree,d);
		tree(pi,1:3) = [bestfeatures(:), bestsplits(:), bestimpurity(:)];
		
        
		% record labels for child nodes
		children = zeros(size([parents;parents]));
        bfpairs = repmat(bestfeatures,2 * (size(parents,2) - 3),1);
        bfpairs = bfpairs(:)';
		pairedindices = sub2ind(size(labels),bfpairs,1:size(labels,2))';
        pairedindices = reshape(pairedindices, size(children,2) - 3, size(children,1) )';
		children(:,4:end) = labels(pairedindices);
		tree = [tree; children];
		
		% update nodes
		bestfeatures = bestfeatures';
		Fv = bestfeatures(n);
		bestsplits = bestsplits';
		Sv = bestsplits(n);
		Iv = sub2ind(size(X),1:size(X,1),Fv');
		Vv = X(Iv);
        Vv = Vv(:); % to column vector
        n = 2*n - (Vv < Sv);
			% Fv(n) = Feature Index
			% Vv(n) = Feature Value
            % Sv(n) = Split Value
    end
%     save tmp1.mat
    
	% update predictions
	leafnodes = getlayer(tree,depth); %Stores the last Layer of tree into LL
    outputlabels = leafnodes(:,4:end); %Stores the predictions of the tree into pp
    p = outputlabels(n,:); %Stores the prediction for each instance into p
    %     p = p(:);
end
