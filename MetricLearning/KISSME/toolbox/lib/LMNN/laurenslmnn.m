function [M, L, Y, C] = lmnn(X, labels)
%LMNN Learns a metric using large-margin nearest neighbor metric learning
%
%   [M, L, Y, C] = lmnn(X, labels)
%
% The function uses large-margin nearest neighbor (LMNN) metric learning to
% learn a metric on the data set specified by the NxD matrix X and the
% corresponding Nx1 vector labels. The metric is returned in M.
%
% 
% (C) Laurens van der Maaten, 2011
% Delft University of Technology


    % Initialize some variables
    [N, D] = size(X);
    assert(length(labels) == N);
    [lablist, ~, labels] = unique(labels);
    K = length(lablist);
    label_matrix = false(N, K);
    label_matrix(sub2ind(size(label_matrix), (1:length(labels))', labels)) = true;
    same_label = logical(double(label_matrix) * double(label_matrix'));
    M = eye(D);
    C = Inf; prev_C = Inf;
    
    % Set learning parameters
    min_iter = 50;          % minimum number of iterations
    max_iter = 1000;        % maximum number of iterations
    eta = .1;               % learning rate
    mu = .5;                % weighting of pull and push terms
    tol = 1e-3;             % tolerance for convergence
    best_C = Inf;           % best error obtained so far
    best_M = M;             % best metric found so far
    no_targets = 3;         % number of target neighbors
    
    % Select target neighbors
    sum_X = sum(X .^ 2, 2);
    DD = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (X * X')));
    DD(~same_label) = Inf; DD(1:N + 1:end) = Inf;
    [~, targets_ind] = sort(DD, 2, 'ascend');
    targets_ind = targets_ind(:,1:no_targets);
    targets = false(N, N);
    targets(sub2ind([N N], vec(repmat((1:N)', [1 no_targets])), vec(targets_ind))) = true;
    
    % Compute pulling term between target neigbhors to initialize gradient
    slack = zeros(N, N, no_targets);        
    G = zeros(D, D);
    for i=1:no_targets
        G = G + (1 - mu) .* (X - X(targets_ind(:,i),:))' * (X - X(targets_ind(:,i),:));
    end
    
    % Perform main learning iterations
    iter = 0;
    while (prev_C - C > tol || iter < min_iter) && iter < max_iter
        disp(iter);
        % Compute pairwise distances under current metric
        sum_X = sum((X * M) .* X, 2);
        DD = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * ((X * M) * X')));
        
        % Compute value of slack variables
        old_slack = slack;
        for i=1:no_targets
            slack(:,:,i) = ~same_label .* max(0, bsxfun(@minus, 1 + DD(sub2ind([N N], (1:N)', targets_ind(:,i))), DD));
        end
        
        % Compute value of cost function
        prev_C = C;
        C = (1 - mu) .* sum(DD(targets)) + ...  % push terms between target neighbors
                 mu  .* sum(slack(:));          % pull terms between impostors
        
        % Maintain best solution found so far (subgradient method)
        if C < best_C
            best_C = C;
            best_M = M;
        end
        
        % Perform gradient update
        for i=1:no_targets
            
            % Add terms for new violations
            [r, c] = find(slack(:,:,i) > 0 & old_slack(:,:,i) == 0);
            G = G + mu .* ((X(r,:) - X(targets_ind(r, i),:))' * ...
                           (X(r,:) - X(targets_ind(r, i),:)) - ...
                           (X(r,:) - X(c,:))' * (X(r,:) - X(c,:)));
            
            % Remove terms for resolved violations
            [r, c] = find(slack(:,:,i) == 0 & old_slack(:,:,i) > 0);
            G = G - mu .* ((X(r,:) - X(targets_ind(r, i),:))' * ...
                           (X(r,:) - X(targets_ind(r, i),:)) - ...
                           (X(r,:) - X(c,:))' * (X(r,:) - X(c,:)));
        end
        M = M - (eta ./ N) .* G;
        
        % Project metric back onto the PSD cone
        [V, L] = eig(M);
        V = real(V); L = real(L);
        ind = find(diag(L) > 0);
        if isempty(ind)
            warning('Projection onto PSD cone failed. All eigenvalues were negative.'); break
        end
        M = V(:,ind) * L(ind, ind) * V(:,ind)';
        if any(isinf(M(:)))
            warning('Projection onto PSD cone failed. Metric contains Inf values.'); break
        end
        if any(isnan(M(:)))
            warning('Projection onto PSD cone failed. Metric contains NaN values.'); break
        end
        
        % Update learning rate
        if prev_C > C
            eta = eta * 1.01;
        else
            eta = eta * .5;
        end
        
        % Print out progress
        iter = iter + 1;
        no_slack = sum(slack(:) > 0);
        if rem(iter, 10) == 0
            %[~, sort_ind] = sort(DD, 2, 'ascend');
            [~, sort_ind]= mink(DD,2);
            disp(['Iteration ' num2str(iter) ': error is ' num2str(C ./ N) ...
                  ', nearest neighbor error is ' num2str(sum(labels(sort_ind(2,:)) ~= labels) ./ N) ...
                  ', number of constraints: ' num2str(no_slack)]);
                  
        end
        global eva;
        if ~isempty(eva),
            valerr=eva(sqrt(L(ind,ind))*V(:,ind)').*100
        end;
    end
    
    % Return best metric and error
    M = best_M;
    C = best_C;
    
    % Compute mapped data
    [L, S, ~] = svd(M);
    L = bsxfun(@times, sqrt(diag(S)), L);
    Y = X * L;
end

function x = vec(x)
    x = x(:);
end
