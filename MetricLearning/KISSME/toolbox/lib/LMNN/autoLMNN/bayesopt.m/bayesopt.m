function [minsample,minvalue,botrace] = bayesopt(F,opt)
    warning('off')
    % Check options for minimum level of validity
    check_opts(opt);

	% Are we doing CBO?
    if isfield(opt,'do_cbo') && opt.do_cbo,
		DO_CBO = true;
	else
		DO_CBO = false;
		con_values = []; % Dummy value, won't actually be used
	end

	if isfield(opt,'optimize_ei') && opt.optimize_ei,
		OPT_EI = true;
	else
		OPT_EI = false;
	end

	if isfield(opt,'ei_burnin'),
		EI_BURN = opt.ei_burnin;
	else
		EI_BURN = 5;
	end
	
	if isfield(opt,'parallel_jobs'),
		PAR_JOBS = opt.parallel_jobs;
	else
		PAR_JOBS = 1;
	end

	if isfield(opt,'parallel_mc_iters'),
		MC_ITERS = opt.parallel_mc_iters;
	else
		MC_ITERS = 5;
	end

    % Draw initial candidate grid from a Sobol sequence
	if isfield(opt,'grid')
		hyper_grid = scale_point(opt.grid,opt.mins,opt.maxes);
		opt.grid_size = size(hyper_grid,1);
	else 
    	sobol = sobolset(opt.dims);
    	hyper_grid = sobol(1:opt.grid_size,:);
		if isfield(opt,'filter_func'), % If the user wants to filter out some candidates
			hyper_grid = scale_point(opt.filter_func(unscale_point(hyper_grid,opt.mins,opt.maxes)),opt.mins,opt.maxes);
		end
	end
    incomplete = logical(ones(size(hyper_grid,1),1));
	
	% Check for existing trace
	if isfield(opt,'resume_trace') && opt.resume_trace && exist(opt.trace_file,'file')
		load(opt.trace_file);
		samples = botrace.samples;
		values = botrace.values;
		times = botrace.times;
		if DO_CBO,
			con_values = botrace.con_values;
		end
		clear botrace
	else
		samples = [];
		values = [];
		con_values = [];
		times = [];
		if isfield(opt,'initial_points'),
			%samples = opt.initial_points;
			for i = 1:size(opt.initial_points,1),
				%fprintf('Running initial point #%d...\n',i);
				init_pt = opt.initial_points(i,:);
				sinit_pt = scale_point(init_pt,opt.mins,opt.maxes);
				if ~DO_CBO,
					vali = F(init_pt);
				else
					[vali,coni] = F(init_pt);
				end
				samples = [samples;sinit_pt];
				values = [values;vali];
				if DO_CBO,
					con_values = [con_values;coni];
				end
			end	
		end
		init = floor(rand(1,2)*opt.grid_size);
		%fprintf('Running first point...\n'); 
	    % Get values for the first two samples (no point using a GP yet)
	    pt1 = unscale_point(hyper_grid(init(1),:),opt.mins,opt.maxes);
	    if ~DO_CBO,
			val1 = F(pt1); % First sample
		else
			[val1,con1] = F(pt1);
		end
	    
		%fprintf('Running second point...\n');
	    pt2 = unscale_point(hyper_grid(init(2),:),opt.mins,opt.maxes);
	    if ~DO_CBO,
			val2 = F(pt2); % Second sample
		else
			[val2,con2] = F(pt2);
   		end
 
		incomplete(init) = false;
	    samples = [samples;hyper_grid(init,:)];
	    values = [values;val1;val2];
		if DO_CBO,
			con_values = [con_values;con1;con2];
		end
	    
	    % Remove first two samples from grid
	    hyper_grid = hyper_grid(incomplete,:);
	    incomplete = logical(ones(size(hyper_grid,1),1));	
	end
    % Main BO loop
	i_start = length(values) - 2 + 1;
    for i = i_start:opt.max_iters-2,
		hidx = -1;
		if PAR_JOBS <= 1,
	        [hyper_cand,hidx,aq_val] = get_next_cand(samples,values,hyper_grid,opt,DO_CBO,con_values,OPT_EI,EI_BURN);
		else
			% Pick first candidate
			[mu_obj,sigma2_obj] = get_posterior(samples,values,hyper_grid,opt,-1);
			
			if ~DO_CBO,
				best = min(values);
				ei = compute_ei(best,mu_obj,sigma2_obj);
			else
				which_feas = all(bsxfun(@le,con_values,opt.lt_const),2);
				best = min(values(which_feas));
				if isempty(best), best = max(values)+999; end
				ei = compute_ei(best,mu_obj,sigma2_obj);
				prFeas = ones(length(ei),1);
				for k = 1:length(opt.lt_const),
					[mu_con,sigma2_con] = get_posterior(samples,con_values(:,k),hyper_grid,opt,-1);
					prFeas = prFeas.*normcdf(repmat(opt.lt_const(k),length(mu_con),1),mu_con,sqrt(sigma2_con));
				end
				ei = prFeas.*ei;
			end

			[mei,meidx] = max(ei);
			hyper_cands = hyper_grid(meidx,:);
						
			% Pick rest of candidates
			for j = 2:PAR_JOBS,
				% Get GP predictive posterior for fantasy candidates
				[mu_f,sigma2_f] = get_posterior(samples,values,hyper_cands,opt,-1);

				mu_con = [];
				sigma2_con = [];
				
				% If we are doing CBO, also get predictive posteriors for constraints for fantasy points
				if DO_CBO,
					for C = 1:length(opt.lt_const),
						[mu_C,sigma2_C] = get_posterior(samples,con_values(:,C),hyper_cands,opt,-1);
						mu_con = [mu_con mu_C];
						sigma2_con = [sigma2_con sigma2_C];
					end
				end	
	
				eiks = {};
				parfor k = 1:MC_ITERS, % MCMC loop to get EI marginalized over fantasy candidates
					% Fantasize objective function values for fantasy candidates
					fant_y = normrnd(mu_f,sqrt(sigma2_f));
					
					% If we aren't doing CBO, that's all we need; marginalize EI over GP predictive posterior using fant_y as the labels.
					if ~DO_CBO,
						[mu,sigma2] = get_posterior([samples;hyper_cands],[values;fant_y],hyper_grid,opt,-1);
						best = min([values;fant_y]);
						ei_k = compute_ei(best,mu,sigma2);
						eiks{k} = ei_k;	
					else % If we are doing CBO, we also need to marginalize prFeas over GP posteriors for the constraints
						[mu_obj,sigma2_obj] = get_posterior([samples;hyper_cands],[values;fant_y],hyper_grid,opt,-1);
						fant_cvals = [];
						
						% Fantasize constraint function values for each fantasy point	
						for C = 1:length(opt.lt_const),
							fant_C = normrnd(mu_con(:,C),sigma2_con(:,C));
							fant_cvals = [fant_cvals;fant_C];
						end
			
						which_feas = all(bsxfun(@le,con_values,opt.lt_const),2);
						which_fant_feas = all(bsxfun(@le,fant_cvals,opt.lt_const),2);
						best = min([values(which_feas);fant_y(which_fant_feas)]);
						if isempty(best), best = max([values;fant_y])+999; end
						
						ei_k = compute_ei(best,mu_obj,sigma2_obj);
							
						prFeas = ones(length(ei_k),1);
						for C = 1:length(opt.lt_const),
							[mu_conf,sigma2_conf] = get_posterior([samples;hyper_cands],[con_values(:,C);fant_cvals(:,C)],hyper_grid,opt,-1);
							prFeas = prFeas.*normcdf(repmat(opt.lt_const(C),length(mu_conf),1),mu_conf,sqrt(sigma2_conf));
						end
							
						ei_k = prFeas .* ei_k;
						eiks{k} = ei_k;
					end	
				end
				ei = mean(cell2mat(eiks),2);
				[mei,meidx] = max(ei);
				hyper_cand = hyper_grid(meidx,:);

        		incomplete(meidx) = false;
	        	hyper_grid = hyper_grid(incomplete,:);
	        	incomplete = logical(ones(size(hyper_grid,1),1));

				hyper_cands = [hyper_cands;hyper_cand];
			end
		end
       	
			
		if PAR_JOBS <= 1,
			if ~DO_CBO, 
        		%fprintf('Iteration %d, ei = %f',i+2,aq_val);
			else
				%fprintf('Iteration %d, eic = %f',i+2,aq_val);
			end
		else
			%fprintf('Iteration %d, running %d jobs in parallel...\n',i+2,PAR_JOBS);
			for k = 1:PAR_JOBS,
				hyper_cands(k,:) = unscale_point(hyper_cands(k,:),opt.mins,opt.maxes);
			end
		end
       
        
        % Evaluate the candidate with the highest EI to get the actual function value, and add this function value and the candidate to our set.
		if ~DO_CBO,
			if PAR_JOBS <= 1,
	        	tic;
				value = F(hyper_cand);
        		times(end+1) = toc;
		        samples = [samples;scale_point(hyper_cand,opt.mins,opt.maxes)];
		        values(end+1) = value;
			else
				par_values = {};
				par_times = {};
				parfor k = 1:PAR_JOBS,
					tic;
					par_values{k} = F(hyper_cands(k,:));
					par_times{k} = toc;
					%fprintf('    * Got value=%f\n',par_values{k});
				end
				for k = 1:PAR_JOBS,
					values = [values;par_values{k}];
					times = [times;par_times{k}];
					samples = [samples;scale_point(hyper_cands(k,:),opt.mins,opt.maxes)];
				end	
			end
		else
			if PAR_JOBS <= 1,
				tic;
				[value,con_value] = F(hyper_cand);
				times(end+1) = toc;
				con_values = [con_values;con_value];
				values(end+1) = value;
				samples = [samples;scale_point(hyper_cand,opt.mins,opt.maxes)];
			else
				par_values = {};
				par_times = {};
				par_con_values = {};
				parfor k = 1:PAR_JOBS,
					tic;
					[v,cv] = F(hyper_cands(k,:));
					par_times{k} = toc;
					par_values{k} = v;
					par_con_values{k} = cv;
					%fprintf('    * Got value=%f, feasible=%d\n',v,all(par_con_values{k}<=opt.lt_const));
				end
				for k = 1:PAR_JOBS,
					values = [values;par_values{k}];
					con_values = [con_values;par_con_values{k}];
					times = [times;par_times{k}];
					samples = [samples;scale_point(hyper_cands(k,:),opt.mins,opt.maxes)];
				end
			end
		end


        % Remove this candidate from the grid (I use the incomplete vector like this because I will use this vector for other purposes in the future.)
        if hidx >= 0,
        	incomplete(hidx) = false;
        	hyper_grid = hyper_grid(incomplete,:);
        	incomplete = logical(ones(size(hyper_grid,1),1));
        end
		
		if PAR_JOBS <= 1,
	        if ~DO_CBO, 
	        	%fprintf(', value = %f, overall min = %f\n',value,min(values));
			else
				which_feas = all(bsxfun(@le,con_values,opt.lt_const),2);
				%fprintf(', value = %f, feasible = %d, overall min = %f\n',value,all(con_value<=opt.lt_const),min(values(which_feas)));
			end
		else
			if ~DO_CBO,
				%fprintf('Overall min = %f\n\n',min(values));
			else
				which_feas = all(bsxfun(@le,con_values,opt.lt_const),2);
				%fprintf('Overall min = %f\n\n',min(values(which_feas)));
			end
		end
		
        botrace.samples = unscale_point(samples,opt.mins,opt.maxes);
        botrace.values = values;
        botrace.times = times;
		if DO_CBO,
			botrace.con_values = con_values;
		end
        if opt.save_trace
            save(opt.trace_file,'botrace');
        end
    end
	
	% Get minvalue and minsample
	if DO_CBO,
		which_feas = all(bsxfun(@le,con_values,opt.lt_const),2);
    	[mv,mi] = min(values(which_feas));
    	minvalue = mv;
		fsamples = samples(which_feas,:);
    	minsample = unscale_point(fsamples(mi,:),opt.mins,opt.maxes);
	else
		[mv,mi] = min(values);
		minvalue = mv;
		minsample = unscale_point(samples(mi,:),opt.mins,opt.maxes);
	end

function [hyper_cand,hidx,aq_val] = get_next_cand(samples,values,hyper_grid,opt,DO_CBO,con_values,OPT_EI,EI_BURN)
        % Get posterior means and variances for all points on the grid.
        [mu,sigma2,ei_hyp] = get_posterior(samples,values,hyper_grid,opt,-1);
        
        % Compute EI for all points in the grid, and find the maximum.
		if ~DO_CBO,
       		best = min(values);
		else
			which_feas = all(bsxfun(@le,con_values,opt.lt_const),2);
			best = min(values(which_feas));
			if isempty(best),
				best = max(values)+999;
			end
		end
        ei = compute_ei(best,mu,sigma2);

        hyps = {};
        ys = {};
        hyps{1} = ei_hyp;
        ys{1} = values;

		if DO_CBO,
			prFeas = ones(length(ei),1);
			for k = 1:length(opt.lt_const),
				[mu_con,sigma2_con,con_hyp] = get_posterior(samples,con_values(:,k),hyper_grid,opt,-1);
				prFeas = prFeas.*normcdf(repmat(opt.lt_const(k),length(mu_con),1),mu_con,sqrt(sigma2_con));
				hyps{k+1} = con_hyp;
				ys{k+1} = con_values(:,k);
			end
			ei = prFeas.*ei;
		end

		if OPT_EI && length(values)>EI_BURN,
			hg_star = zeros(size(hyper_grid));
			if ~DO_CBO,
				parfor k = 1:length(hyper_grid),
					z = hyper_grid(k,:);
					zstar = optimize_ei(z,samples,values,best,hyps{1},opt);
					hg_star(k,:) = max(zstar,0);
				end
			else
				parfor k = 1:length(hyper_grid),
					z = hyper_grid(k,:);
					zstar = optimize_eic(z,samples,ys,best,hyps,opt);
					hg_star(k,:) = max(zstar,0);
				end
			end

        	[mu,sigma2,ei_hyp] = get_posterior(samples,values,hg_star,opt,-1);
	        ei = compute_ei(best,mu,sigma2);
			if DO_CBO,
				prFeas = ones(length(ei),1);
				for k = 1:length(opt.lt_const),
					[mu_con,sigma2_con,con_hyp] = get_posterior(samples,con_values(:,k),hg_star,opt,-1);
					prFeas = prFeas.*normcdf(repmat(opt.lt_const(k),length(mu_con),1),mu_con,sqrt(sigma2_con));
				end
				ei = prFeas.*ei;
			end

			[mei,meidx] = max(ei);
			hyper_cand = unscale_point(hg_star(meidx,:),opt.mins,opt.maxes);
			hidx = -1;
		else
    		[mei,meidx] = max(ei);
        	hyper_cand = unscale_point(hyper_grid(meidx,:),opt.mins,opt.maxes);
        	hidx = meidx;
        end
		
		aq_val = mei;

function [mu,sigma2,hyp] = get_posterior(X,y,x_hats,opt,hyp)
    meanfunc = opt.meanfunc;
    covfunc = opt.covfunc;
    if isnumeric(hyp)
        if isfield(opt,'num_mean_hypers'),
            n_mh = opt.num_mean_hypers;
        else
            n_mh = num_hypers(meanfunc{1},opt);
        end
        if isfield(opt,'num_cov_hypers'),
            n_ch = opt.num_cov_hypers;
        else
            n_ch = num_hypers(covfunc{1},opt);
        end
        hyp = [];
        hyp.mean = zeros(n_mh,1);
        hyp.cov = zeros(n_ch,1);
        hyp.lik = log(0.1);
		hyp = minimize(hyp,@gp,-100,@infExact,meanfunc,covfunc,@likGauss,X,y);
    end
    [mu,sigma2] = gp(hyp,@infExact,meanfunc,covfunc,@likGauss,X,y,x_hats);

function zstar = optimize_ei(z,X,y,best,hyp,opt)

	if isfield(opt,'cov_grad_f'),
		cov_grad_f = opt.cov_grad_f;
	else
		cov_grad_f = @covSEard_grad;
	end
	covfunc = opt.covfunc{:};
	meanfunc = opt.meanfunc{:};

	K = covfunc(hyp.cov,X);
	Q = K + exp(2*hyp.lik)*eye(size(K));
	prior_mean = meanfunc(hyp.mean,X);
	k_hat = covfunc(hyp.cov,X,z);


	Qiy = linsolve(Q,y - prior_mean);
	Qik = linsolve(Q,k_hat);

	zstar = minimize(z,@EI_F,-50,X,Qiy,Qik,cov_grad_f,best,hyp,opt);

function zstar = optimize_eic(z,X,ys,best,hyps,opt)
	if isfield(opt,'cov_grad_f'),
		cov_grad_f = opt.cov_grad_f;
	else
		cov_grad_f = @covSEard_grad;
	end
	covfunc = opt.covfunc{:};
	meanfunc = opt.meanfunc{:};

	Qiys = {};
	Qiks = {};

	for j = 1:length(hyps),
		Kj = covfunc(hyps{j}.cov,X);
		Q = Kj + exp(2*hyps{j}.lik)*eye(size(Kj));
		prior_mean = meanfunc(hyps{j}.mean,X);
		k_hat = covfunc(hyps{j}.cov,X,z);
		y = ys{j};

		Qiys{j} = linsolve(Q,y - prior_mean);
		Qiks{j} = linsolve(Q,k_hat);
	end

	zstar = minimize(z,@EIC_F,-50,X,Qiys,Qiks,cov_grad_f,best,hyps,opt);


function [ei,ei_grad] = EI_F(z,X,Qiy,Qik,cov_grad_f,best,hyp,opt)
	if nargout > 1,
		[ei,ei_grad] = ei_obj_grad(z,X,Qiy,Qik,cov_grad_f,best,hyp,opt);
		ei = -ei;
		ei_grad = -ei_grad;
	else
		ei = ei_obj_grad(z,X,Qiy,Qik,cov_grad_f,best,hyp,opt);
		ei = -ei;
	end

function [eic,eic_grad] = EIC_F(z,X,Qiys,Qiks,cov_grad_f,best,hyps,opt)
	if nargout > 1,
		[eic,eic_grad] = eic_obj_grad(z,X,Qiys,Qiks,cov_grad_f,best,hyps,opt);
		eic = -eic;
		eic_grad = -eic_grad;
	else
		eic = eic_obj_grad(z,X,Qiys,Qiks,cov_grad_f,best,hyps,opt);
		eic = -eic;
	end


% Computes EI(z;best) and dEI(z;best)/dz
function [ei,ei_grad] = ei_obj_grad(z,X,Qiy,Qik,cov_grad_f,best,hyp,opt)
	covfunc = opt.covfunc{:};
	meanfunc = opt.meanfunc{:};
	
	prior_mean = meanfunc(hyp.mean,z);
	k_hat = covfunc(hyp.cov,X,z);

	% Compute GP predictive posterior
	mu = prior_mean + k_hat'*Qiy;
	sigma2 = covfunc(hyp.cov,z,'diag') - k_hat'*Qik;
	sigma2 = max(sigma2,1e-10);

	sigma = sqrt(sigma2);
	u = (best - mu) ./ sigma;
	ucdf = normcdf(u);
    updf = normpdf(u);
    ei = sigma .* (u.*ucdf + updf);

	if nargout > 1,
		dk_dx = cov_grad_f(hyp.cov,X,z);
		mu_grad = dk_dx'*Qiy;
		s2_grad = (cov_grad_f(hyp.cov,z,z) - 2*Qik'*dk_dx)';

		dEI_dmu = -ucdf;
    	dEI_ds2 = updf ./ (2*sigma);

    	ei_grad = dEI_dmu.*mu_grad + dEI_ds2.*s2_grad;
	end

function [pf,pf_grad] = pf_obj_grad(z,X,Qiy,Qik,cov_grad_f,lambda,hyp,opt)
	covfunc = opt.covfunc{:};
	meanfunc = opt.meanfunc{:};

	prior_mean = meanfunc(hyp.mean,z);
	k_hat = covfunc(hyp.cov,X,z);
	
	mu = prior_mean + k_hat'*Qiy;
	sigma2 = covfunc(hyp.cov,z,'diag') - k_hat'*Qik;
	sigma = sqrt(sigma2);

	pf = normcdf(lambda,mu,sigma);

	if nargout > 1,
		dk_dx = cov_grad_f(hyp.cov,X,z);
		mu_grad = dk_dx'*Qiy;
		s2_grad = (cov_grad_f(hyp.cov,z,z) - 2*Qik'*dk_dx)';

		Z = (lambda - mu) ./ sigma;
		pf_grad = -normpdf(Z)*(mu_grad./sigma + s2_grad/(2*sigma2)*Z);
	end

function [eic,eic_grad] = eic_obj_grad(z,X,Qiys,Qiks,cov_grad_f,best,hyps,opt)
	if nargout > 1,
		[ei,ei_grad] = ei_obj_grad(z,X,Qiys{1},Qiks{1},cov_grad_f,best,hyps{1},opt);
	else
		ei = ei_obj_grad(z,X,Qiys{1},Qiks{1},cov_grad_f,best,hyps{1},opt);
	end

	pfs = zeros(length(opt.lt_const),1);

	if nargout > 1,
		pf_grads = zeros(length(opt.lt_const),length(z));
	end

	for j = 1:length(opt.lt_const),
		Qiy = Qiys{j+1};
		Qik = Qiks{j+1};
		hyp = hyps{j+1};
		lambda = opt.lt_const(j);

		if nargout > 1,
			[pf,pf_grad] = pf_obj_grad(z,X,Qiy,Qik,cov_grad_f,lambda,hyp,opt);
		else
			[pf,~] = pf_obj_grad(z,X,Qiy,Qik,cov_grad_f,lambda,hyp,opt);	
		end	

		pfs(j) = pf;

		if nargout > 1,
			pf_grads(j,:) = pf_grad;
		end
	end

	eic = ei .* prod(pfs);

	if nargout > 1,
		eic_grad = ei_grad .* prod(pfs);
		for j = 1:length(opt.lt_const),
			gradterm = (ei .* pf_grads(j,:) .* (prod(pfs)/pfs(j)))';
			eic_grad = eic_grad + gradterm;
		end
	end


function ei = compute_ei(best,mu,sigma2)
    sigmas = sqrt(sigma2);
    u = (best - mu) ./ sigmas;
    ucdf = normcdf(u);
    updf = normpdf(u);
    ei = sigmas .* (u .* ucdf + updf);


% Returns the derivative of covSEard w.r.t. a single candidate z
function [k] = covSEard_grad(hyp,x,z)
	D = length(z);
	ell = exp(hyp(1:D));
	sf2 = exp(2*hyp(D+1)); 

	k = covSEard(hyp,x,z);
	sq_der = (diag(-2./(ell.^2))*(bsxfun(@minus,x,z))')';
	k = -0.5*bsxfun(@times,k,sq_der);
 
function upt = unscale_point(x,mins,maxes)
    if size(x,1) == 1,
        upt = x .* (maxes - mins) + mins;
    else
        upt = bsxfun(@plus,bsxfun(@times,x,(maxes-mins)),mins);
    end

function pt = scale_point(x,mins,maxes)
	pt = bsxfun(@rdivide,bsxfun(@minus,x,mins),maxes-mins);
    
function check_opts(opt)
    if ~isfield(opt,'dims')
        error('bayesopt:opterror',['The dims option specifying the dimensionality of ' ...
            'the optimization problem is required']);
    end
    
    if ~isfield(opt,'mins') || length(opt.mins) < opt.dims
        error('bayesopt:opterror','Must specify minimum values for each hyperparameter');
    end
       
    if ~isfield(opt,'maxes') || length(opt.maxes) < opt.dims
        error('bayesopt:opterror','Must specify maximum values for each hyperparameter');
    end 

	if isfield(opt,'parallel_jobs') && isfield(opt,'optimize_ei') && opt.parallel_jobs > 1,
		error('bayesopt:opterror','Parallel jobs with optimize_ei on is not supported yet.');
	end

	if isfield(opt,'optimize_ei') && opt.optimize_ei && ~isfield(opt,'cov_grad_f'),
		warning('bayesopt:optwarning','Warning: opt.optimize_ei is set, but opt.cov_grad_f is not. By default, covSEard is assumed.');
	end
    
function nh = num_hypers(func,opt)
    str = func();
    nm = str2num(str);
    if ~isempty(nm)
        nh = nm;
    else
        if all(str == '(D+1)')
            nh = opt.dims + 1;
        elseif all(str == '(D+2)')
            nh = opt.dims + 2;
        else
            error('bayesopt:unkhyp','Unknown number of hyperparameters asked for by one of the functions');
        end
    end
    
