%% Initializing the Gaussian Process (GP) Library
clc;
addpath ../
addpath ../gpml/
startup;


%% We define the function we would like to optimize
% We want to minimize the first output but ensure the 2nd output is <-0.05
b = 1; % Pass in constant parameters (like datasets) to the function handle.
F = @(X) samplef(X(1),X(2),b); % CBO needs a function handle whose sole parameter is a vector of the parameters to optimize over.

% Let's plot it just to see what we are trying to optimize
clf;
[xx,yy]=meshgrid(-5:0.05:0,-5:0.05:5);
[zz,cc]=samplef(xx(:),yy(:),1);
zz(cc>0.05)=NaN;
surf(xx,yy,reshape(zz,size(xx)));
drawnow;

%% Setting parameters for Bayesian Global Optimization
opt = defaultopt(); % Get some default values for non problem-specific options.
opt.dims = 2; % Number of parameters.
opt.mins = [-5,-5]; % Minimum value for each of the parameters. Should be 1-by-opt.dims
opt.maxes = [0, 5]; % Vector of maximum values for each parameter. 
opt.max_iters = 50; % Override the default max_iters value -- probably don't need 100 for this simple demo function.
opt.grid_size = 20000;
%opt.parallel_jobs = 3; % Run 3 jobs in parallel using the approach in (Snoek et al., 2012). Increases overhead of BO, so probably not needed for this simple function.
opt.lt_const = -0.05;
%opt.optimize_ei = 1; % Uncomment this to optimize EI/EIC at each candidate rather than optimize over a discrete grid. This will be slow.
%opt.grid_size = 300; % If you use the optimize_ei option
opt.do_cbo = 1; % Do CBO -- use the constraint output from F as well.
%opt.save_trace = 1;
%opt.trace_file = 'demo_trace.mat';
%matlabpool 3; % Uncomment to do certain things in parallel. Suggested if optimize_ei is turned on. If parallel_jobs is > 1, bayesopt does this for you.

%% Start the optimization
fprintf('Optimizing hyperparamters of function "samplef.m" ...\n');
[ms,mv,T] = bayesopt(F,opt);   % ms - Best parameter setting found
                               % mv - best function value for that setting L(ms)
                               % T  - Trace of all settings tried, their function values, and constraint values.
                              
%% Print results
fprintf('******************************************************\n');
fprintf('Best hyperparameters:      P1=%2.4f, P2=%2.4f\n',ms(1),ms(2));
fprintf('Associated function value: F([P1,P2])=%2.4f\n',mv);
fprintf('******************************************************\n');

%% Draw optimium
hold on;
plot3([ms(1) ms(1)],[ms(2) ms(2)],[max(zz) min(zz)],'r-','LineWidth',2);



