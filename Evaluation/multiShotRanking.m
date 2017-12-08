function [dis,param]=multiShotRanking(probeFeat,probeId,galleryFeat,galleryId,evalType,metric,data)

rankType=evalType.rankType;

if(strcmp(data.evalType,'clustering') && ~strcmp(data.name,'DukeMTMC'))
    nClusters=data.numClusters;
    % Cluster test features
    [probeFeat,probeId]=clusterFeatures1(probeFeat,probeId,nClusters);
    [galleryFeat,galleryId]=clusterFeatures1(galleryFeat,galleryId,nClusters);
end

if(strcmp(data.name,'DukeMTMC'))
end

% for other methods, features are already projected
switch metric.name
    case 'kissme'        
        probeFeat=probeFeat*metric.T;
        galleryFeat=galleryFeat*metric.T;  
    case 'xqda'
        probeFeat=probeFeat*metric.T.W;
        galleryFeat=galleryFeat*metric.T.W;
end

% rank

uniqueProbeTestingId=unique(probeId,'stable');
uniqueGalleryTestingId=unique(galleryId,'stable');
dis=zeros(length(uniqueProbeTestingId),length(uniqueGalleryTestingId));
switch rankType
    case {'rnp','ahisd','sanp'}        
        if(strcmp(rankType,'sanp'))
            param.lam=1e-2;
            param.w1=1e-1;
            param.w2=1;
            param.pcaFactor=0.5;
        elseif(strcmp(rankType,'rnp'))
            param.lam1=1e-1; 
            param.lam2=1e-1;
        elseif(strcmp(rankType,'ahisd'))
            param.pcaFactor=0.5;
        end
        for i=1:length(uniqueProbeTestingId)            
            Xp=probeFeat(find(probeId==uniqueProbeTestingId(i)),:);            
            for j=1:length(uniqueGalleryTestingId)
                Xg=galleryFeat(find(galleryId==uniqueGalleryTestingId(j)),:);
                dis(i,j)=computeLinearHullDistance(Xp',Xg',rankType,param);
            end
        end
    case 'srid'
        dis=srid(galleryFeat',galleryId,probeFeat',probeId);
        param=[];
    case 'isr'
        dis=isr(galleryFeat',galleryId,probeFeat',probeId);
        param=[];
    case 'afda'
        dis=afda(galleryFeat',galleryId,probeFeat',probeID);
        param=[];
end

function [f,nids]=clusterFeatures1(feat,ids,nClusters)

uids=unique(ids,'stable');
f=[];nids=[];
for i=1:length(uids)
    currId=uids(i);
    ids1=find(ids==currId);
    currFeat=feat(ids1,:);
    % Cluster features and retrieve cluster centers
    if(size(currFeat,1)>nClusters)
        [C,~]=vl_kmeans(currFeat',nClusters);
        C=C';
    else
        C=currFeat;
    end
    f=[f;C];
    nids=[nids currId*ones(1,size(C,1))];
end

function dis=isr(galleryFeatures,galleryId,probeFeatures,probeId)

param.pos    = true; 
param.mode   = 2;    
param.lambda = 0.9;  
nIter = Inf;

idPersons=unique(galleryId);
cmcCurrent = zeros(length(idPersons),3);
cmcCurrent(:,1) = 1:length(idPersons);
cmc = zeros(length(idPersons),3);
cmc(:,1) = 1:length(idPersons);
[Dnorm,~] = normalizeBase(galleryFeatures);

[featuresTest featuresTestNorms] = normalizeBase(probeFeatures);
Alphas = full( mexLasso(single(featuresTest),single(Dnorm),param) );

% [~, col] = find(isnan(Alphas));
% Alphas(:,col)=[];
% featuresTest(:,col)=[];
% probeId(col)=[];

uniqueLabel=unique(probeId);
tmp=uniqueLabel;

for j=1:length(uniqueLabel)
    %j
    currId=uniqueLabel(j);
  idxSamePerson=find(probeId==currId);
  % Select the feature vector of that group
  eachfeaturesTest = featuresTest(:,idxSamePerson);
  % Select the reconstructed coefficients
  Alpha = Alphas(:,idxSamePerson);
  % Check if reconstruction worked
  checkSparsityError

  %% Perform iterative re-weighted ranking
  [errorSort finalLabel] = ...
  sparseClassifyIterateWeighted_demo(eachfeaturesTest,Dnorm,Alpha,...
                                     param,idPersons,galleryId,nIter,...
                                     length(idxSamePerson));
  %% Evaluation
  [cmc cmcCurrent] = evaluateCMC_demo(currId,finalLabel...
                                      ,cmc,cmcCurrent);
end
dis=cmcCurrent(:,2)./cmcCurrent(:,3);

function dis=srid(galleryFeatures,galleryId,probeFeatures,probeId)


% Form groups vector
groups=[];
uniqueGalleryId=unique(galleryId,'stable');
uniqueProbeId=unique(probeId,'stable');
assert(isequal(uniqueGalleryId,uniqueProbeId));
for i=1:length(uniqueGalleryId)
    groups(i)=length(find(galleryId==uniqueGalleryId(i)));
end

for i=1:length(uniqueProbeId)
    curr_id_imgs=find(probeId==uniqueProbeId(i));
    resNet{i}=zeros(1,length(uniqueProbeId));
    for j=curr_id_imgs
        obs=probeFeatures(:,j);
        [x,e]=groupSparsityADM_again(galleryFeatures,obs,groups);        
        % Compute residuals and find arg min residuals
        m=0;res=[];res1=[];
        ids=[];
        for k=uniqueProbeId
            m=m+1;xt=zeros(length(x),1);
            xt(find(galleryId==k))=x(find(galleryId==k));
            res(m)=norm(obs-galleryFeatures*xt-e);ids(m)=k;
        end       
        % Add up residuals
        resNet{i} = resNet{i}+res;%+res1;            
    end
end
for i=1:length(uniqueProbeId)
    for j=1:length(uniqueGalleryId)
        dis(i,j)=resNet{i}(j);
    end
end

% ADM for group sparsity
% Ref: Deng, Yin, and Zhang, Group Sparse Optimization by Alternating Direction Method

function [x,e]=groupSparsityADM_again(A,b,groups)

% Initialize 
[m,n]=size(A);AtA=A'*A;At=A';Atb=A'*b;
z=zeros(n,1);lam1=z;lam2=zeros(m,1);e=zeros(length(b),1);
gamma1=0.1;gamma2=gamma1;beta1=(2*m)/norm(b,1);beta2=beta1;
iter=2;shrink=@(x,lam)sign(x).*max(abs(x)-lam,0);

for i=1:iter
    tmp1=(beta1*eye(n)+beta2*AtA);
    tmp2=(beta1*z-lam1+beta2*Atb+At*lam2-beta2*A'*e);
    x=tmp1\tmp2;
    e=shrink(-A*x+b+lam2/beta2,1/beta2);
    z=groupShrink(x,lam1,beta1,groups);
    lam1=lam1-gamma1*beta1*(z-x);
    lam2=lam2-gamma2*beta2*(A*x-b+e);
end

function z=groupShrink(x,lam,beta,groups)

z=[];s=length(groups);t=1;
for i=1:s
    tmp=x(t:t+groups(i)-1)+(1/beta)*lam(t:t+groups(i)-1);
    z(t:t+groups(i)-1)=max(norm(tmp)-(1/beta),0)*tmp/norm(tmp);
    t=t+groups(i);
end
z=z';



function d=computeLinearHullDistance(X1,X2,algo,param)

if(strcmp(algo,'ahisd'))
    % Constructs separating hyperplane given two hulls and determines the
    % euclidean distance between them
    % Basically, solve equation (4) in Cevikalp and Triggs, Face Recognition 
    % Based on Image Sets, CVPR 2010. See the discussion following equation
    % (4), (6) in this paper.
    H1=constructHull(X1,param.pcaFactor);H2=constructHull(X2,param.pcaFactor);
    U1=H1.basis;U2=H2.basis;
    m1=H1.mean;m2=H2.mean;

    U=orth([U1 -U2]);
    c1=m1-U*(U'*m1);
    c2=m2-U*(U'*m2);
    w=c1-c2;
    d=norm(w);
elseif(strcmp(algo,'rnp'))
    [x,y]=findClosestHullPoints(X1,X2,algo,param);
    d=norm(x-y);
elseif(strcmp(algo,'sanp'))
    [x,y,U1,m1,v1,U2,m2,v2,coef1,coef2]=findClosestHullPoints(X1,X2,algo,param);
    w1=param.w1;w2=param.w2;
    d=(w1*norm((m1+U1*v1)-(m2+U2*v2))^2+...
            w2*norm(m1+U1*v1-X1*coef1)^2+...
            w2*norm(m2+U2*v2-X2*coef2)^2)*(length(v1)+length(v2));
end


% Given two sets X and Y, find two points x and y, one in the affine hull
% of each set, such that x and y are the closest. 

function[x,y,varargout]=findClosestHullPoints(X,Y,algo,param)

if(strcmp(algo,'ahisd'))
    % Construct affine hulls for each set X and Y
    H1=constructHull(X,param.pcaFactor);
    H2=constructHull(Y);

    % Find closest points on the affine hulls

    U1=H1.basis;U2=H2.basis;
    m1=H1.mean;m2=H2.mean;

    U=orth([U1 -U2]);
    v=inv(U'*U)*U'*(m2-m1);
    v1=v(1:size(U1,2));v2=v(size(U1,2)+1:end);
    x=m1+U1*v1;y=m2+U2*v2;
elseif(strcmp(algo,'rnp'))
    % Instead, find closest points on the affine hull using the regularized
    % nearest points algorithm, Yang et al., Face recognition based on
    % regularized nearest points between image sets
    [x,y]=findClosestPointsRNP(X,Y,param);
elseif(strcmp(algo,'sanp'))
    if(nargout==2)
        [x,y]=findClosestPointsSANP(X,Y,param);
    else
        [x,y,U1,m1,v1,U2,m2,v2,coef1,coef2]=findClosestPointsSANP(X,Y,param);
        varargout{1}=U1;varargout{2}=m1;varargout{3}=v1;
        varargout{4}=U2;varargout{5}=m2;varargout{6}=v2;
        varargout{7}=coef1;varargout{8}=coef2;
    end    
end

x=x/norm(x);y=y/norm(y);

function [x1,x2,varargout]=findClosestPointsSANP(X1,X2,param)

[U1,m1]=pcaBasis(X1,param.pcaFactor);[U2,m2]=pcaBasis(X2,param.pcaFactor);
lambda=param.lam;opt.w1=param.w1;opt.w2=param.w2;

[v1,v2,coef1,coef2,~,~] = accel_grad_solver(X1,m1,U1,X2,m2,U2,...
            lambda*max(abs((2*opt.w2).*(X1'*m1))),lambda*max(abs((2*opt.w2).*(X2'*m2))),opt);
x1=m1+U1*v1;x2=m2+U2*v2;
if(nargout>2)
    varargout{1}=U1;varargout{2}=m1;varargout{3}=v1;
    varargout{4}=U2;varargout{5}=m2;varargout{6}=v2;
    varargout{7}=coef1;varargout{8}=coef2;
end
        

function [U,m]=pcaBasis(X,pcaFactor)

[U, m, vars] = pca(X);
ind = find(cumsum(vars) / sum(vars) > pcaFactor, 1, 'first');
%U=U(:, 1:ind);


function [x1,x2]=findClosestPointsRNP(X1,X2,param)

lam1=param.lam1;
lam2=param.lam2;
X1  =  X1./( repmat(sqrt(sum(X1.*X1)), [size(X1,1),1]) );
X2  =  X2./( repmat(sqrt(sum(X2.*X2)), [size(X2,1),1]) );

% Find sum of singular values of X1 and X2 -- see eq (11) in Yang et al.,
% Face recognition based on regularized nearest points between image sets.
s1=sumSingularValues(X1);s2=sumSingularValues(X2);

newy = [zeros(size(X1,1),1); 1;1];
tem1 = [ones(1,size(X1,2)) zeros(1,size(X2,2))];
tem2 = [zeros(1,size(X1,2)) sqrt(lam1/lam2)*ones(1,size(X2,2))];
newD = [[X1 -sqrt(lam1/lam2)*X2];tem1;tem2];

DD            =  newD'*newD;
Dy            =  newD'*newy;
x             =  (DD+lam1*eye(size(newD,2)))\Dy;

x1      =X1* x(1:size(X1,2),1)*sqrt(s1+s2);
x2      =X2*sqrt(lam1/lam2)*x(size(X1,2)+1:end,1)*sqrt(s1+s2);

function s = sumSingularValues(X)

var=svd(X);
s=sum(var);

% Given a set of features X for a specific person, construct a hull -
% supports affine hull only currently

function H=constructHull(X,pcaFactor)

% Represent affine hull using the mean vector m and the orthonormal basis U
% for the directions spanned by the affine subspace - refer section 2.1 in
% Cevikalp and Triggs, Face Recognition Based on Image Sets, CVPR 2010.

% Compute mean
m=mean(X')';

% Center X and compute SVD
Xc=X-repmat(m, [1 size(X,2)]);
[U,S,~]=svd(Xc'*Xc);
[val,ind]=sort(diag(S),'descend');
U=U(:,ind);

% Find dominant directions
dDirections=find((cumsum(val)/sum(val))>=pcaFactor); %0.8 for clusetering+fda

% Required basis
Ub=Xc*U(:,1:dDirections(1));
%Ub=Xc*U;

% Orthonormalize
for i=1:size(Ub,2)
    Ub(:,i)=Ub(:,i)/norm(Ub(:,i));
end

H.mean=m;
H.basis=Ub;

% September 29, 2010
% written by Yiqun Hu

% This function implements the accelerated gradient algorithm for 
% multi-task learning regularized by trace norm as
% described in Ji and Ye (ICML 2009).

% References:
%Ji, S. and Ye, J. 2009. An accelerated gradient method for trace norm minimization. 
%In Proceedings of the 26th Annual international Conference on Machine Learning 
%(Montreal, Quebec, Canada, June 14 - 18, 2009). ICML '09, vol. 382. ACM, New York, 
%NY, 457-464.

%[Wp, fval_vec,itr_counter] =
%accel_grad_mtl(Xtrain,Ytrain,lambda,opt)

% required inputs:
% Xtrain: K x 1 cell in which each cell is N x D where N is the sample size
% and D is data dimensionality and K is the number of tasks
% Ytrain: K x 1 cell in which each cell contains the output of the
% corresponding task
% lambda: regularization parameter

% optional inputs:
% opt.L0: Initial guess for the Lipschitz constant
% opt.gamma: the multiplicative factor for Lipschitz constant
% opt.W_init: initial weight matrix
% opt.epsilon: precision for termination
% opt.max_itr: maximum number of iterations

% outputs:
% Wp: the computed weight matrix
% fval_vec: a vector for the sequence of function values
% itr_counter: number of iterations executed


function [new_v1,new_v2,new_coef1,new_coef2,fval_vec,itr_counter] = accel_grad_solver(X1,mu1,U1,X2,mu2,U2,lambda1,lambda2,opt)

if nargin<8
    opt = [];
end

if isfield(opt, 'L0')
    L0 = opt.L0;
else
    L0 = 100;
end

if isfield(opt, 'gamma')
    gamma = opt.gamma;
else
    gamma = 1.1;
end

if isfield(opt, 'w1')
    w1 = opt.w1;
else
    w1 = 1.0;
end

if isfield(opt, 'w2')
    w2 = opt.w2;
else
    w2 = 1.0;
end

if isfield(opt, 'v1_init')
    v1_init = opt.v1_init;    
else
    v1_init = zeros(size(U1,2),1);
end
if isfield(opt, 'v2_init')
    v2_init = opt.v2_init;    
else
    v2_init = zeros(size(U2,2),1);
end
if isfield(opt, 'coef1_init')
    coef1_init = opt.coef1_init;    
else
    coef1_init = zeros(size(X1,2),1);
end
if isfield(opt, 'coef2_init')
    coef2_init = opt.coef2_init;    
else
    coef2_init = zeros(size(X2,2),1);
end

if isfield(opt, 'epsilon')
    epsilon = opt.epsilon;
else
    epsilon = 1.0e-5;
end

if isfield(opt, 'max_itr')
    max_itr = opt.max_itr;
else
    max_itr = 100;
end

factors{1,1} = (2*w1+2*w2).*U1'*U1;
factors{1,2} = (2*w1).*U1'*U2;
factors{1,3} = (2*w1).*U1'*(mu2-mu1);
factors{1,4} = (2*w2).*U1'*mu1;
factors{1,5} = (2*w2).*U1'*X1;

factors{2,1} = (2*w1+2*w2).*U2'*U2;
factors{2,2} = (2*w1).*U2'*U1;
factors{2,3} = (2*w1).*U2'*(mu2-mu1);
factors{2,4} = (2*w2).*U2'*mu2;
factors{2,5} = (2*w2).*U2'*X2;

factors{3,1} = (2*w2).*X1'*X1;
factors{3,2} = (2*w2).*X1'*mu1;
factors{3,3} = (2*w2).*X1'*U1;

factors{4,1} = (2*w2).*X2'*X2;
factors{4,2} = (2*w2).*X2'*mu2;
factors{4,3} = (2*w2).*X2'*U2;

v1_old = v1_init;
v2_old = v2_init;
coef1_old = coef1_init;
coef2_old = coef2_init;
alpha = 1;
fval_vec = [];
L = L0;
fval_old = -inf;
fval = 0;
itr_counter = 0;
Z1_old = v1_old;
Z2_old = v2_old;
Z3_old = coef1_old;
Z4_old = coef2_old;
while abs(fval_old-fval)>epsilon
    fval_old = fval;
    [new_v1,new_v2,new_coef1,new_coef2,P] = ComputeQP(X1,mu1,U1,X2,mu2,U2,Z1_old,Z2_old,Z3_old,Z4_old,L,w1,w2,lambda1,lambda2,factors);
    f = w1*norm((mu1+U1*new_v1)-(mu2+U2*new_v2))^2;
    f = f + w2*norm(mu1+U1*new_v1-X1*new_coef1)^2;
    f = f + w2*norm(mu2+U2*new_v2-X2*new_coef2)^2;
    fval = f+lambda1*norm(new_coef1,1)+lambda2*norm(new_coef2,1);
    Q = P+lambda1*norm(new_coef1,1)+lambda2*norm(new_coef2,1);
    while fval>Q
        %fprintf('Searching step size (fval = %f, Q = %f)...\n',fval,Q);
        L = L*gamma;
        [new_v1,new_v2,new_coef1,new_coef2,P] = ComputeQP(X1,mu1,U1,X2,mu2,U2,Z1_old,Z2_old,Z3_old,Z4_old,L,w1,w2,lambda1,lambda2,factors);
        f = w1*norm((mu1+U1*new_v1)-(mu2+U2*new_v2))^2;
        f = f + w2*norm(mu1+U1*new_v1-X1*new_coef1)^2;
        f = f + w2*norm(mu2+U2*new_v2-X2*new_coef2)^2;
        fval = f+lambda1*norm(new_coef1,1)+lambda2*norm(new_coef2,1);
        Q = P+lambda1*norm(new_coef1,1)+lambda2*norm(new_coef2,1);
    end
    
    alpha_old = alpha;
    alpha = (1+sqrt(1+4*alpha_old^2))/2;
    Z1_old = new_v1+((alpha_old-1)/alpha)*(new_v1-v1_old);
    Z2_old = new_v2+((alpha_old-1)/alpha)*(new_v2-v2_old);
    Z3_old = new_coef1+((alpha_old-1)/alpha)*(new_coef1-coef1_old);
    Z4_old = new_coef2+((alpha_old-1)/alpha)*(new_coef2-coef2_old);
        
    v1_old = new_v1;
    v2_old = new_v2;
    coef1_old = new_coef1;
    coef2_old = new_coef2;
    fval_vec = [fval_vec,fval];
    itr_counter = itr_counter+1;
    if itr_counter>max_itr
        break;
    end
%     if mod(itr_counter,100)==0
%         fprintf('Iteration = %8d,  objective = %f\n',itr_counter, fval);
%     end
    
end
return;

function [new_v1,new_v2,new_coef1,new_coef2,P] = ComputeQP(X1,mu1,U1,X2,mu2,U2,v1,v2,coef1,coef2,L,w1,w2,lambda1,lambda2,factors)

[aux_v1,aux_v2,aux_coef1,aux_coef2,delta] = ComputeGradStep(X1,mu1,U1,X2,mu2,U2,v1,v2,coef1,coef2,L,w1,w2,factors);
new_v1 = aux_v1;
new_v2 = aux_v2;
zero_idx = find(abs(aux_coef1)-lambda1/L<=0);
new_coef1 = sign(aux_coef1).*(abs(aux_coef1)-lambda1/L);
new_coef1(zero_idx) = 0;
zero_idx = find(abs(aux_coef2)-lambda2/L<=0);
new_coef2 = sign(aux_coef2).*(abs(aux_coef2)-lambda2/L);
new_coef2(zero_idx) = 0;

P = w1*norm((mu1+U1*v1)-(mu2+U2*v2))^2 + w2*norm(mu1+U1*v1-X1*coef1)^2 + w2*norm(mu2+U2*v2-X2*coef2)^2;
diff_vec = [new_v1-v1;new_v2-v2;new_coef1-coef1;new_coef2-coef2];
P = P+delta'*diff_vec+0.5*L*norm(diff_vec)^2;
return;

function [aux_v1,aux_v2,aux_coef1,aux_coef2,delta] = ComputeGradStep(X1,mu1,U1,X2,mu2,U2,v1,v2,coef1,coef2,L,w1,w2,factors)
delta = ComputeDerivative(X1,mu1,U1,X2,mu2,U2,v1,v2,coef1,coef2,w1,w2,factors);
aux_v1 = v1 - (1/L)*delta(1:length(v1));
aux_v2 = v2 - (1/L)*delta(length(v1)+1:length(v1)+length(v2));
aux_coef1 = coef1 - (1/L)*delta(length(v1)+length(v2)+1:length(v1)+length(v2)+length(coef1));
aux_coef2 = coef2 - (1/L)*delta(length(v1)+length(v2)+length(coef1)+1:end);
return;

function dev = ComputeDerivative(X1,mu1,U1,X2,mu2,U2,v1,v2,coef1,coef2,w1,w2,factors)

dev = [factors{1,1}*v1-factors{1,2}*v2-factors{1,3}+factors{1,4}-factors{1,5}*coef1];
dev = [dev;factors{2,1}*v2-factors{2,2}*v1+factors{2,3}+factors{2,4}-factors{2,5}*coef2];
dev = [dev;factors{3,1}*coef1-factors{3,2}-factors{3,3}*v1];
dev = [dev;factors{4,1}*coef2-factors{4,2}-factors{4,3}*v2];
return;












