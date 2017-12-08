function embedding=gb_lmnn(X,Y,K,L,varargin)
% Nonlinear metric learning using gradient boosting regression trees.
% 'X' (dxn) is the input training data, 'labels' (1xn) the corresponing labels
% 'L' (kxd) is an initial linear transformation which can be learned using LMNN
% L corresponds to a metric M=L'*L
% 'model' corresponds to the nonlinear mapping function, which include the
% linear part - L, and the nonlinear part - ensemble of trees
% 'pred' is the transformed training data
% gb_lmnn_v1.m: only consider small number of potential imposters
%
% $Revision: 144 $
% $Date: 2013-09-17 14:07:49 -0500 (Tue, 17 Sep 2013) $ 
%

[un,~,labels]=unique(Y);
options.classes=length(un);
options.K=K;                 % number of nearest neighbours
options.tol = 1e-3;         % tolerance for convergence
options.verbose=true;   % screen output
options.depth=4;           % tree depth
options.ntrees=200;      % number of boosted trees
options.lr=1e-3;             % learning rate for gradient boosting
options.no_potential_impo=50;
options.buildlayer = @buildlayer_sqrimpurity_openmp_multi;
options.XVAL=[];
options.YVAL=[];
options=extractpars(varargin,options);

pred=L*X;
if ~isempty(options.XVAL), % define validiation criterion for early stopping
    predVAL=L*options.XVAL;
    computevalerr=@(pred,predVAL) knncl([],pred, Y,predVAL,options.YVAL,1,'train',0); 
else,
    predVAL=[];
    computevalerr=@(pred,predVAL) -1.0;
end;
    
% Initialize some variables
[D,N] = size(X);
assert(length(labels) == N);

% find K target neighbors
targets_ind=findtargetneighbors(X,labels,options);

% sort the training input feature-wise (column-wise)
N = size(X,2);
[Xs,Xi] = sort(X');

% initialize ensemble (cell array of trees)
ensemble{1}=[];
ensemble{2}={};

% initialize the lowest validation error
lowestval=inf;
embedding=@(xTr) xTr;

% initialize roll-back in case stepsize is too large
OC=inf;
Opred=pred;
OpredVAL=predVAL;

iter=0;
% Perform main learning iterations
while(length(ensemble{1})<=options.ntrees)
    % Select potential imposters
    if  ~rem(iter, 10)        
        active=findimpostors(pred,labels,options);
        OC=inf; % allow objective to go up
    end
    [hinge,grad]=lmnnobj(pred,int16(targets_ind'),int16(active));
    C=sum(hinge);

    if C>OC, % roll back in case things go wrong
      C=OC;
      pred=Opred;
      predVAL=OpredVAL;
      % remove from ensemble
      ensemble{1}(end)=[];
      ensemble{2}(end)=[];
      if options.verbose,fprintf('Learing rate too large (%2.4e) ...\n',options.lr);end;
      options.lr=options.lr/2.0;
    else, % otherwise increase learning rate a little
      options.lr=options.lr*1.01;
    end;

    % Perform gradient boosting: construct trees to minimize loss
    [tree,p] = buildtree(X',Xs,Xi,-grad',options.depth,options);
    
    % update predictions and ensemble
    Opred=pred;
    OC=C;
    OpredVAL=predVAL;
    pred = pred + options.lr * p'; % update predictions
	iter=length(ensemble{1})+1;
    ensemble{1}(iter) = options.lr; % add learning rate to ensemble
    ensemble{2}{iter} = tree; % add tree to ensemble
    
    % update embeding of validation data
    if ~isempty(options.XVAL), predVAL=predVAL+options.lr*evaltree(options.XVAL',tree)';end;
            
            
    % Print out progress
    no_slack = sum(hinge > 0);
    if (~rem(iter, 5)||iter==1) && options.verbose
        disp(['Iteration ' num2str(iter) ': loss is ' num2str(C ./ N) ...
		       ', violating inputs: ' num2str(no_slack) ', learning rate: ' num2str(options.lr)]);
    end

    if mod(iter,10)==0 || iter==options.ntrees,
    model.L=L;
    model.ensemble=ensemble;
    newemb=@(xTr) evalensemble(xTr',model.ensemble,xTr'*model.L')';    
    valerr=computevalerr(pred,predVAL);
        if valerr<=lowestval,
            lowestval=valerr;
            embedding=newemb;
            if options.verbose & lowestval>=0.0,fprintf('Best validation error: %2.2f%%\n', lowestval*100.0);end;
        end;
    end;
end



function x = vec(x);
x = x(:);


function targets_ind=findtargetneighbors(X,labels,options);
[D,N]=size(X);
targets_ind=zeros(N,options.K);
for i=1:options.classes
    u=i;
    jj=find(labels==u);
    Xu=X(:,jj);
    T=buildmtreemex(Xu,50);
    targets=usemtreemex(Xu,Xu,T,options.K+1);
    targets_ind(jj,:)=jj(targets(2:end,:))';
end;
    
    
function active=findimpostors(pred,labels,options);
[~,N]=size(pred);
active=zeros(options.no_potential_impo,N);
for i=1:options.classes
    ii=find(labels==i);
    pi=pred(:,ii);
    jj=find(labels~=i);
    pj=pred(:,jj);
    Tj=buildmtreemex(pj,50);
    active(:,ii)=jj(usemtreemex(pi,pj,Tj,options.no_potential_impo));
end;
    
