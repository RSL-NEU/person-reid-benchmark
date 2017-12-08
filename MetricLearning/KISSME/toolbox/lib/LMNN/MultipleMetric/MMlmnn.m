function [Lb,Det]=MMlmnn(X,Y,Kg,varargin)
% function [Lb,Det]=MMlmnn(X,Y,Kg,'param1',val1,'param2',val2,...);  
%                          
% Input:
%
% X  = Input data dxn with d dimensions and n input vectors
% Y  = Labels 1xn (must be integers)
% Kg = Number of neighbors used (recommended 1 or 3)
%                                  
% Possible parameters:
% maxiter (default=1000)  Maximum number of iterations
% stepsize (default=1e-07) Set initial stepsize
% weight1 (default=0.5) Weight of first term in objective                                
% verbose (default=1) Write out status messages (0=silent, 1=informative,2=chatty)
% initl (default=[]) Provide initial matrix for all classes (if M=L'L, you must pass on L!!!)
% thresha (default=1e-15) Terminate if stepsize is below this value
% thresho (default=1e-05) Terminate if relative objective improvement is below this value
% maximp (default=100000) Limit number of impostors per update 
% diag (default=0) Restrict to diagonal matrices
% validation (default=0) Set this fraction as validation data set
% classsplit (default=0) Split the validation data per class (ie if validation is 0.1 take 10% of EACH CLASS)                                           
% fixe (default=[]) A 1xn vector that allocates different metrics to each data point (default makes it identical to 1 per class)

% Advanced features:                        
%
% noatlas (default=0) Switches off blas / lapack
% initialmb (default=[]) Initializes the Mahalanobis M=L'L, one for each metric
% initiallb (default=[]) Initializes one matrix L for each metric
% checkupcountdown (default=10) Find all impostors after that many iterations
% valinline (default=1) Compute the validation inline (if set to 0 an external function is called)
% valcounter (default=10) Computer validation error after that many steps
% avg (default=0.8) The gradient step is avg* the average gradient and (1-avg) times the actual gradient (some kind of regularization)
% stepup (default=1.01) If an improvement is made multiply stepsize by this factor 
% stepdown (default=0.5) If no improvement is made multiply stepsize by this factor
% minstepsize (default=1e-10) Stepsize can never fall below this threshold
% finish (default=0) For internal use only (sets the state to the final convergence mode)
%                                                               

% Output:
% Lb       cell array of matrices, one for each class
% Det      Set of statistis about run
%
% =======================
% copyright by Kilian Q. Weinberger 2005
% kilianweinberger@yahoo.de
%


% Make the most important variables global
global E Mb Lb x y Lx un Lm Ln un;
x=X;y=Y; Lx=[];Lb=[]; Mb=[]; un=[]; E=[];
Ln=size(x,1);
Lm=Ln;
clear('X','Y');
if(nargin<3)
    Kg=3;
end;


% backwards compatibility issues
if(length(varargin)>1 & isnumeric(varargin{1}))
 for i=2:length(varargin)
  varargin{i-1}=varargin{i};
 end;
end;


un=unique(y);

tic;

% extract dimension and number of input vectors
[dim,n]=size(x);

% extract input parameters
pars.weight1=0.5;
pars.noatlas=0;
pars.verbose=1;
pars.initialmb=[];
pars.initiallb=[];
pars.initl=[];
pars.stepsize=1e-07;
pars.thresha=1e-15;
pars.thresho=1e-05;
pars.fixe=[];
pars.finish=0;
pars.checkupcountdown=10;
pars.maximp=100000;
pars.validation=0;
pars.classsplit=0;
pars.valinline=1;
pars.valcounter=10;
pars.diag=0;
pars.avg=0.8;
pars.stepup=1.01;
pars.stepdown=0.5;
pars.minstepsize=1e-10;
pars.maxiter=1000;
pars=extractpars(varargin,pars);




% Prepare validation data set
if(pars.validation<0 | pars.validation>1) 
  error('Please keep validation factor between 0 and 1. \n');
end;


% fix originalE assignment as one per class
originalE=zeros(1,length(y));
for i=1:length(unique(y))
  j=find(y==un(i));
  originalE(j)=i;
end;

% split the data for validation purposes
[itr,ite]=makesplits(y,1-pars.validation,1,pars.classsplit,Kg+1);
xva=x(:,ite);
yva=y(:,ite);
x=x(:,itr);
y=y(itr);
n=length(itr);

% in case the assignment is pre-specified overwrite it
if(~isempty(pars.fixe)) 
	originalE=pars.fixe; 
end;
E=originalE(itr);	
unE=unique(originalE);
d=length(unE);	% specify how many metrics there are

% initialize some variables
besterr=inf;
bestLb=Lb;
bestiter=1;

if(pars.diag)
 statement('Restricted to diagonal matrices\n',0,pars.verbose);
end;

if(pars.validation>0 & ~pars.valinline)
 touch .reset
 delete .waiting
 delete .feedback
 delete .enough
end;




% Initialize basis matrices
if(~isempty(pars.initl))
    for i=1:d
        pars.initiallb(:,i)=pars.initl(:);
    end;
    [Lm,Ln]=size(pars.initl);
else
    if(isstruct(pars.initiallb))
        Lm=pars.initiallb.Lm;
        Ln=pars.initiallb.Ln;
        pars.initiallb=pars.initiallb.Lb;
    else
        if(~isempty(pars.initiallb))
            Lm=round(sqrt(size(pars.initiallb(:,1),1)));
            Ln=Lm;
        else
            Lm=size(x,1);
            ln=size(x,1);
        end;
    end;
end;

x=x(1:Ln,:);
xva=xva(1:Ln,:);

if(~isempty(pars.initiallb)) 
 for i=1:d
  L=reshape(pars.initiallb(:,i),Lm,Ln);
  Mb(:,i)=vec(L'*L);
 end;
else
    Mb=pars.initialmb;
end;
if(isempty(Mb)) Mb=repmat(vec(eye(size(x,1))),1,d);end;
updateLb(pars);
checkupcounter=0;
finishup=0;


% compute nearest neighbors of each vector
[gen,NN]=getGenLS(x,y,Kg,pars);

% compute distance to nearest neighbors
Ni=computeDistancesAS(repmat(1:n,Kg,1),NN)+1;

% first part of gradient (attract Kg-NN)
dfG=SOPE2(gen(1,:),gen(2,:),pars);
dF=dfG.*pars.weight1;

% Find all impostors 
updateLb(pars);
imp=checkup(Kg,Ni,pars);


% Create list buffer (pure optimization)
for nnid=1:Kg; a1{nnid}=[];a2{nnid}=[];end;

% Main Loop   <----------------------------------------
for iter=1:pars.maxiter
 statement(sprintf('Iter: %i:  ',iter),0,pars.verbose);

 % save backup information
 for nnid=1:Kg; a1old{nnid}=a1{nnid};a2old{nnid}=a2{nnid};end;
 dFold=dF;


 % Initialize Counters
 impostors=0;
 dF2=zeros(size(dF));
 
 % Only if there are potenetial impostors
 if(size(imp,2)>0)
  % Compute distances to impostors
  g0=computeDistancesAS(imp(1,:),imp(2,:));
  g1=computeDistancesAS(imp(2,:),imp(1,:));
  % Compute distances to nearest neighbors 
  Ni=computeDistancesAS(repmat(1:n,Kg,1),NN)+1;
  
  % Rotate through the Kg-neighborhood
  for k=1:Kg
    % See which impostors are active
    active1=find(g0<=Ni(k,imp(1,:)));
    active2=find(g1<=Ni(k,imp(2,:)));
%    active2=[];
    
    % update buffer
    if(~isempty(a1{k}) | ~isempty(a2{k}))
     [plus1,minus1]=sd(active1(:)',a1{k}(:)');
     [plus2,minus2]=sd(active2(:)',a2{k}(:)');
    else
     plus1=active1;plus2=active2;
     minus1=[];minus2=[];
    end;

   
   % Sort distances as additive and subtractive 
   PLUS1=[        imp(1,plus1)          imp(2,plus2)  imp(1,minus1) imp(2,minus2)];
   PLUS2=[NN(k,imp(1,plus1)) NN(k,imp(2,plus2)) imp(2,minus1) imp(1,minus2)];

   MINUS1=[    imp(1,minus1)              imp(2,minus2)  imp(1,plus1) imp(2,plus2)];
   MINUS2=[NN(k,imp(1,minus1)) NN(k,imp(2,minus2)) imp(2,plus1) imp(1,plus2)];

   % update derivative
    %%%%dF=dF+SOPE2(PLUS1,PLUS2)-SOPE2(MINUS1,MINUS2); BTTF
    dF2=dF2+SOPE2(PLUS1,PLUS2,pars)-SOPE2(MINUS1,MINUS2,pars);
% Update objective function
    impostors=impostors+length(active1)+length(active2);
    a1{k}=active1;a2{k}=active2;    
  end;
 end	
   dF=dF+dF2.*(1-pars.weight1);
   obj=sum(sum(dF.*Mb,1),2)+impostors*(1-pars.weight1);

%   dF=dF+pars.reg*(sign(Mb-II));  % EXPERIMENTAL L1 REGULARIZATION
%   dFreg=2.*pars.reg*(Mb-II);  % EXPERIMENTAL L2 REGULARIZATION
%   reg=pars.reg.*sum(sum((Mb-II).^2));
%   obj=obj+reg;
  if(obj<0)
   fprintf('Objective is negative! This can never happen. There must be bug!\n');
   fprintf('Or it is very very small \n');
   break;
  end;

  % Print status
  if(iter>1) objdiff=obj-Det.obj(iter-1);else objdiff=0;end;
  statement(sprintf('Obj:%2.4f Imps:%i   stepsize:%i  diff:%i ',obj,impostors,pars.stepsize,objdiff),0, ...
	    pars.verbose);
  if(obj==0)
    statement('Obj is 0000',1,pars.verbose);
  end;
 Det.obj(iter)=obj;
 Det.imp(iter)=impostors;
 
  % take a gradient step  
 if(checkupcounter==0 | Det.obj(iter-1)>Det.obj(iter)-(1e-05)) 
% FOR DEBUGGING
statement(sprintf(' %f*Average ',pars.avg),1,pars.verbose);
dFav=repmat(sum(dF,2),1,size(dF,2));
%

  % save old state
  Mbold=Mb;
  
%  Mb=Mb-pars.stepsize.*(dFav+dFreg);		% gradient step
  Mb=Mb-pars.stepsize.*(pars.avg.*dFav+(1-pars.avg).*dF);		% gradient step
  updateLb(pars);				% update variables

                                        % still psd
  statement('',1,pars.verbose);		% print out
  pars.stepsize=pars.stepsize*pars.stepup;	% increase stepsize by 1%
 else
   % if last step increased objective, back-up and decrease
   % stepsize 

   pars.stepsize=pars.stepsize*pars.stepdown;  
  if(finishup)
   statement('**^^**',1,pars.verbose);      
   Mb=Mb-pars.stepsize.*dF;
   updateLb(pars);
   pars.checkupcountdown=max(round(pars.checkupcountdown*0.98),10);
   checkupcounter=min(checkupcounter,pars.checkupcountdown);   
  else 
   statement('**STALLED**',1,pars.verbose);
   for nnid=1:Kg; a1{nnid}=a1old{nnid};a2{nnid}=a2old{nnid};end; 
   dF=dFold;	    
   Det.obj(iter)=Det.obj(iter)+1e-05;
   pars.checkupcountdown=max(round(pars.checkupcountdown/2),10);
   checkupcounter=min(checkupcounter,pars.checkupcountdown);   
  end;
 end;
 if(pars.stepsize<pars.minstepsize) pars.stepsize=pars.minstepsize;end;
 
 % Terminate if stepsize is too small
 if(pars.stepsize<pars.thresha & checkupcounter~=0) 
   if(checkupcounter>5 & checkupcounter<pars.checkupcountdown-5) 
	 checkupcounter=pars.checkupcountdown;
   else
    statement('\nStepsize too small!',1,pars.verbose);
    if(finishup==1 | ~pars.finish) break; else pars.stepsize=1e-09;finishup=1;statement('\n',1,pars.verbose);end;
   end;
 end;

 % Terminate if progress is too small
 if(iter>20 & abs(mean(diff(Det.obj(iter-10:iter))))<pars.thresho*Det.obj(iter) & checkupcounter~=0) 
   if(checkupcounter>5 & checkupcounter<pars.checkupcountdown-5) 
	 checkupcounter=pars.checkupcountdown;
   else
    statement('\nNo more progress!',1,pars.verbose);
    if(finishup==1 | ~pars.finish) break; else pars.stepsize=1e-09;finishup=1;statement('\n',1,pars.verbose);end;
   end;
 end;

 
 % Every now and again check for new impostors
 if(checkupcounter==pars.checkupcountdown)
statement(sprintf('\n'),1,pars.verbose);
   li=length(imp);
   Ni=computeDistancesAS(repmat(1:n,Kg,1),NN)+1;
   imp2=checkup(Kg,Ni,pars);
   imp2=setdiff(imp2',imp','rows')';   
   [imp i1 i2]=unique([imp imp2].','rows');
   imp=imp.';
   if(size(imp,2)~=li)
     for nnid=1:Kg;
	   a1{nnid}=i2(a1{nnid});
	   a2{nnid}=i2(a2{nnid});
     end;
   end;
   statement(sprintf('Iteration:%i Objective:%f\n',iter,obj),1,pars.verbose);
   statement(sprintf('Added %i new constraints',length(imp)-li),1,pars.verbose);
   
   
   % Flush the saved gradient
   if(length(imp)==li)
    pars.checkupcountdown=min(200,pars.checkupcountdown*2);  
   end;
   checkupcounter=0;
  else
  checkupcounter=checkupcounter+1;  
 end;

 % communicate with validator (if active)
 if(pars.validation>0)
  if(pars.valinline) [bestLb,besterr,enough,bestiter]=inlineval(x,y,xva,yva,Lb,iter,bestLb,besterr,bestiter,pars);end;
  if(~pars.valinline) [bestLb,besterr,enough,bestiter]=outlineval(x,y,xva,yva,Lb,iter,bestLb,besterr,bestiter,pars);end;
  if(enough) break;end;
 end;

end;
% End of Main Loop   <----------------------------------------

% If validation has been set
if(pars.validation>0 & exist('bestLb'))
 Lb=bestLb;
 for i=1:size(Lb,2)
  L=reshape(Lb(:,i),Lm,Ln);
  Mb(:,i)=vec(L'*L);
 end;
 updateLb(pars);
else
 bestiter=iter;
end;

%Output cluster assignments
Det.E=E;
Det.Mb=Mb;
Det.bestiter=bestiter;

% make struct out of Lb to encode if it is rectangular
Lb=createstruct;
Lb.E=originalE;

% Add running time to output
Det.time=toc;
statement(sprintf('\n'),1,pars.verbose);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Lstruct=createstruct
% function Lstruct=createstruct
% 
% creates correct output structure for multiple matrices
global Lb Lm Ln E;

unE=unique(E);
Lstruct.E=E;
Lstruct.un=unE;
for i=1:size(Lb,2);
    Lstruct.L{i}=reshape(Lb(:,i),Lm,Ln);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function  [bestLb,besterr,enough,bestiter]=outlineval(x,y,xva,yva,Lb,iter,bestLb,besterr,bestiter,pars);
% [bestLb,besterr,enough]=function outlineval(x,y,xva,yva,Lb,iter,bestLb,besterr);
%
% validation with satellite thread

  if(exist('./.waiting')==2 & exist('./.feedback')~=2 )
	valID=iter;
    save('mlmnnval','x','y','xva','yva','Lb','valID');
    delete .waiting
  end;
  if(exist('./.feedback')==2)
    fprintf('\nReceiving best matrix ...'); 
	try
     load('mlmnnbest.mat','bestLb','besterr','bestid');
	 bestiter=bestid;
     delete ./.feedback	 
	 fprintf('done Best Error:%2.2f\n',besterr*100);	 
    catch
	 disp(lasterr);
    end;
  end;  
  if(exist('./.enough')==2)
    delete .enough
	fprintf('\nNo more improvement on validation data set!\n');
    enough=1;
  else
   enough=0;
 end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [bestLb,besterr,enough,bestiter]=inlineval(x,y,xva,yva,Lb,iter,bestLb,besterr,bestiter,pars);
% [bestLb,besterr,enough]=function outlineval(x,y,xva,yva,Lb,iter,bestLb,besterr,pars);
%
% validate internally
global un Lm Ln;
persistent bestcounter checkcounter;


if(besterr==inf) bestcounter=0;checkcounter=0;besterr=1000;end;

checkcounter=checkcounter+1;
enough=0;
if(checkcounter<pars.valcounter)  return;end;

checkcounter=0;
statement(sprintf('\n'),1,pars.verbose);
Lstruct=createstruct;
err=MMknnclassify(Lstruct,x,y,xva,yva,3,'verbose',pars.verbose);
statement(sprintf('\r'),1,pars.verbose);
statement(sprintf('Iteration: %i Training error:%2.2f, Validation error:%2.2f best error:%2.2f',iter,err(1)*100,err(2)*100,besterr*100),1,pars.verbose);
err=err(2);
if(err<=besterr)
 besterr=err;
 bestLb=Lb;
 bestcounter=0;
 bestiter=iter;
 statement(sprintf('**'),1,pars.verbose);
else
 bestcounter=bestcounter+1;
% fprintf('(%i/%i)\n',bestcounter,pars.valcounter);
 statement(sprintf('(%i/%i)\n',bestcounter,pars.valcounter),1,pars.verbose); 
 if(bestcounter==pars.valcounter) enough=1;end;
end;
statement(sprintf('\n'),1,pars.verbose);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function imp=curbimp(imp,pars);
% function imp=curbimp(imp,pars);
%
% subsamples pars.maximp constraints
%
  
if(length(imp)>pars.maximp)
 i=randperm(size(imp,2));     % puts constraints in random order
 imp=imp(:,i(1:pars.maximp));   % only uses the first pars.maximp of them
end;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function imp=checkup(Kg,Ni,pars);
% function imp=checkup(Kg,Ni,pars);
%
% returns all pairs of points that do not have matching labels but
% one is an impostore amongst the Kg-neighborhood of the other
%
global Lb x y un;


mats=size(Lb,2);
imp=[];
for c=un(1:end)
 j=find(y~=c);
 i=find(y==c);
 statement(sprintf('Finding impostors for class %i ...',c),1,pars.verbose);
 limps= LSImps(i,j,Ni(end,i),pars); % Find impostors  

 limps=curbimp(limps,pars);  
 imp=[imp limps];
end;
imp=unique(sort(imp)','rows')';
imp=curbimp(imp,pars);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function limps=LSImps(a,b,Thresh,pars);
% function limps=LSImps(a,b,Thresh,pars);
% finds impostors amongst b within the Threshold distance of a
%
global x y Lb E Lx;
B=750;
N1=size(a,2);
N2=size(b,2);
d=size(Lb,2); % number of Lap vectors


limps=[];
for i=1:B:N1
  BB=min(B,N1-i);ind=i:i+BB;
  imp=[];
  for ma=1:size(Lb,2)
    j=find(E(b)==ma);
    if(~isempty(j))
if(pars.noatlas)
    DM=distance(Lx{ma}(:,b(j)),Lx{ma}(:,a(ind)));
    imp=findlessh(DM,Thresh(ind));
    [bi,aj]=ind2sub([length(j),N1],imp);
    newlimps=[bi;aj];
    %newlimps=[b(j(bi));a(aj+i-1)];    
else
    newlimps=findimps3m(Lx{ma}(:,b(j)),Lx{ma}(:,a(ind)), Thresh(ind));
end;
    if(~isempty(newlimps))
     if(newlimps(end)==0)    
      [minv,endpoint]=min(min(newlimps));
      newlimps=newlimps(:,1:endpoint-1);
     end;
     newlimps(1,:)=b(j(newlimps(1,:)));
     newlimps(2,:)=a(newlimps(2,:)+i-1);
	end;
    limps=[limps flipud(newlimps)];
   end;
  end;
  statement(sprintf('(%i) ',round((i+BB)/12*100)),0,pars.verbose);  
end;
statement(sprintf(' [%i] ',size(limps,2)),1,pars.verbose);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function updateLb(pars)
% assigns the global variable Lb to the square-roots of the
% matrices in Mb

global Mb Lb Lx Lm Ln;
Lb=zeros(Lm*Ln,size(Mb,2));
for i=1:size(Mb,2)  
 M=mat(Mb(:,i));
 if(pars.diag)
  dm=max(diag(M),0);
  Lb(:,i)=vec(diag(sqrt(dm)));
  Mb(:,i)=vec(diag(dm));
 else
  [v,d]=eig(M);
  di=diag(d);
  % the max(di,0) line ensures L'*L to be PSD. This might be
  % over restrictive, however it avoids complex numbers.
  L=(v*diag(sqrt(max(di,0))));		 % Project onto psd cone
  dip=abs(di);
  L=(v*diag(sqrt(dip).*sign(di)));		
 % if(min(di)<0) statement('NEGATIVE EIGENVALUE',1,2);end; 

  [temp ind]=sort(-di); 
  L=L(1:Ln,ind(1:Lm))';
  Lb(:,i)=vec(L);
  Mb(:,i)=vec(L'*L);
 end;
end;

global x y;
for i=1:size(Mb,2)    
 Lx{i}=reshape(Lb(:,i),Lm,Ln)*x;
end;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dis=computeDistancesAS(a,b)
% compute the distances between the indices a,b of x under 
% Mahalanobis matrix Mb
% 1. compute distance with every Mahalanobis matrix
% 2. weight them according to the corresponding row in E
% 3. sum up results
global Lb x y E Lx;

[m,n]=size(a);
a=a(:)';b=b(:)';
d=size(Lb,2);
dis=zeros(size(a));
 Eb=E(b);


for i=1:d
  j=find(Eb==i);
%  dis(j)=cdist(mat(Lb(:,i))*x,a(j),b(j));
  dis(j)=cdist(Lx{i},a(j),b(j));
end;
dis=reshape(dis,[m,n]);




  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function op=SOPE2(a,b,pars)
% computes weighted sum of outer products

global E x Mb;

if(pars.diag)
 outersum=@(x,a,b) vec(SODd(x,a,b));
else
 outersum=@(x,a,b) vec(SOD(x,a,b));
end;

mats=size(Mb,2);
[D,N]=size(x);
B=round(2500/D^2*1000000);
op=zeros(D^2,mats);
for i=1:B:length(a)
 BB=min(B,length(a)-i);
 ind=i:i+BB;
 for ma=1:mats
    % add positive weights
    jj=find(E(b(ind))==ma);
    op(:,ma)=op(:,ma)+outersum(x,a(ind(jj)),b(ind(jj)));
 end; 
end;

  




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [gen,NN]=getGenLS(x,y,Kg,pars)
global un;

statement('Computing nearest neighbors ... ',1,pars.verbose);

[D,N]=size(x);
Gnn=zeros(Kg,N);
for c=un
 statement(sprintf('%i nearest genuine neighbors for class %i:',Kg,c),0,pars.verbose);
 i=find(y==c);
 nn=LSKnn(x(:,i),x(:,i),2:Kg+1,pars);
 Gnn(:,i)=i(nn);
 statement('',1,pars.verbose);
end;

statement('',1,pars.verbose);
NN=Gnn;
gen1=vec(Gnn(1:Kg,:)')';
gen2=vec(repmat(1:N,Kg,1)')';
gen=[gen1;gen2];
statement('                                             ',0,pars.verbose);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function NN=LSKnn(X1,X2,ks,pars)
% find nearest neighbors in squared Euclidean distance
% 
B=1000;
[D,N]=size(X2);
NN=zeros(length(ks),N);
DD=zeros(length(ks),N);

for i=1:B:N
  BB=min(B,N-i);
  statement('.',1,pars.verbose);
  DistM=distance(X1,X2(:,i:i+BB));
  statement('.',0,pars.verbose);
  [dist,nn]=sort(DistM);
  clear('DistM');
  statement('.',0,pars.verbose);  
  NN(:,i:i+BB)=nn(ks,:);
  clear('nn','dist');
  statement(sprintf('(%i) ',round((i+BB)/N*100)),1,pars.verbose);    
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function statement(s,newline,verbose)
% prints a statement with an optional new line character

if(verbose==0) return;end;
fprintf(s);
if(newline)
  if(verbose==2) fprintf('\n');
  else
     ss=repmat(' ',1,10-length(s));
     fprintf([ ss '\r']);
  end;
  if(newline==3 & verbose==1) fprintf('\n');end;
end;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function v=vec(M)
v=M(:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function M=mat(v)

s=round(sqrt(length(v)));
M=zeros(s);
M(:)=v(:);

