function [eigvector, eigvalue, elapse] = MFA_CDeng(gnd, options, data) 
% MFA: Marginal Fisher Analysis 
% 
%       [eigvector, eigvalue] = MFA(gnd, options, data) 
%  
%             Input: 
%               data       - Data matrix. Each row vector of fea is a data point. 
% 
%               gnd     - Label vector.   
% 
%               options - Struct value in Matlab. The fields in options 
%                         that can be set: 
%                 intraK         = 0   
%                                     Sc: 
%                                       Put an edge between two nodes if and 
%                                       only if they belong to same class.  
%                                > 0  Sc: 
%                                       Put an edge between two nodes if 
%                                       they belong to same class and they 
%                                       are among the intraK nearst neighbors of 
%                                       each other in this class.   
%                 interK         = 0  Sp: 
%                                       Put an edge between two nodes if and 
%                                       only if they belong to different classes.  
%                                > 0 
%                                     Sp: 
%                                       Put an edge between two nodes if 
%                                       they rank top interK pairs of all the 
%                                       distance pair of samples belong to 
%                                       different classes  
% 
%                         Please see LGE.m for other options. 
% 
%             Output: 
%               eigvector - Each column is an embedding function, for a new 
%                           data point (row vector) x,  y = x*eigvector 
%                           will be the embedding result of x. 
%               eigvalue  - The eigvalue of LPP eigen-problem. sorted from 
%                           smallest to largest.  
%               elapse    - Time spent on different steps  
%  
% 
%    Examples: 
% 
%        
%        
%       fea = rand(50,70); 
%       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4]; 
%       options = []; 
%       options.intraK = 5; 
%       options.interK = 40; 
%       options.Regu = 1; 
%       [eigvector, eigvalue] = MFA(gnd, options, fea); 
%       Y = fea*eigvector; 
%  
%  
% 
% See also LPP, LGE 
% 
%Reference: 
% 
%   S. Yan, D. Xu, B. Zhang, H.-J. Zhang, Q. Yang, and S. Lin. 
%    "Graph embedding and extension: A general framework for 
%    dimensionality reduction." IEEE Transactions on PAMI, 
%    29(1), 2007.  
% 
%   Deng Cai, Xiaofei He, Yuxiao Hu, Jiawei Han, and Thomas Huang,  
%   "Learning a Spatially Smooth Subspace for Face Recognition", CVPR'2007 
% 
%   version 2.1 --June/2007  
%   version 2.0 --May/2007  
%   version 1.1 --Sep/2006  
% 
%   Written by Deng Cai (dengcai2 AT cs.uiuc.edu) 
% 
 
bGlobal = 0; 
if ~exist('data','var') 
    bGlobal = 1; 
    global data; 
end 
 
if (~exist('options','var')) 
   options = []; 
end 
 
 
[nSmp,nFea] = size(data); 
if length(gnd) ~= nSmp 
    error('gnd and data mismatch!'); 
end 
 
intraK = 5; 
if isfield(options,'intraK')  
    intraK = options.intraK; 
end 
 
interK = 20; 
if isfield(options,'interK')  
    interK = options.interK; 
end 
 
tmp_T = cputime; 
 
Label = unique(gnd); 
nLabel = length(Label); 
 
 
D = EuDist2(data,[],0); 
 
nIntraPair = 0; 
if intraK > 0 
    G = zeros(nSmp*(intraK+1),3); 
    idNow = 0; 
    for i=1:nLabel 
        classIdx = find(gnd==Label(i)); 
        DClass = D(classIdx,classIdx); 
        [dump idx] = sort(DClass,2); % sort each row 
        clear DClass dump; 
        nClassNow = length(classIdx); 
        nIntraPair = nIntraPair + nClassNow^2; 
        if intraK < nClassNow 
            idx = idx(:,1:intraK+1); 
        else 
            idx = [idx repmat(idx(:,end),1,intraK+1-nClassNow)]; 
        end 
 
        nSmpClass = length(classIdx)*(intraK+1); 
        G(idNow+1:nSmpClass+idNow,1) = repmat(classIdx,[intraK+1,1]); 
        G(idNow+1:nSmpClass+idNow,2) = classIdx(idx(:)); 
        G(idNow+1:nSmpClass+idNow,3) = 1; 
        idNow = idNow+nSmpClass; 
        clear idx 
    end 
    Sc = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp); 
    [I,J,V] = find(Sc); 
    Sc = sparse(I,J,1,nSmp,nSmp); 
    Sc = max(Sc,Sc'); 
    clear G 
else 
    Sc = zeros(nSmp,nSmp); 
    for i=1:nLabel 
        classIdx = find(gnd==Label(i)); 
        nClassNow = length(classIdx); 
        nIntraPair = nIntraPair + nClassNow^2; 
        Sc(classIdx,classIdx) = 1; 
    end 
end 
 
 
if interK > 0 & (interK < (nSmp^2 - nIntraPair)) 
    maxD = max(max(D))+100; 
    for i=1:nLabel 
        classIdx = find(gnd==Label(i)); 
        D(classIdx,classIdx) = maxD; 
    end 
     
    [dump,idx] = sort(D(:)); 
    clear dump D 
    idx = idx(1:interK); 
    [I, J] = ind2sub([nSmp,nSmp],idx); 
    Sp = sparse(I,J,1,nSmp,nSmp); 
    Sp = max(Sp,Sp'); 
else 
    Sp = ones(nSmp,nSmp); 
    for i=1:nLabel 
        classIdx = find(gnd==Label(i)); 
        Sp(classIdx,classIdx) = 0; 
    end 
end 
 
Dp = full(sum(Sp,2)); 
Sp = -Sp; 
for i=1:size(Sp,1) 
    Sp(i,i) = Sp(i,i) + Dp(i); 
end 
 
Dc = full(sum(Sc,2)); 
Sc = -Sc; 
for i=1:size(Sc,1) 
    Sc(i,i) = Sc(i,i) + Dc(i); 
end 
 
timeW = cputime - tmp_T; 
 
%========================== 
% If data is too large, the following centering codes can be commented 
%========================== 
if isfield(options,'keepMean') & options.keepMean 
    ; 
else 
    if issparse(data) 
        data = full(data); 
    end 
    sampleMean = mean(data); 
    data = (data - repmat(sampleMean,nSmp,1)); 
end 
%========================== 
 
%========================== 
% Sc is not guaranteed to be non-singular, we have to keep less principle 
% components. A better way might be using regularization instead of PCA 
% 
% options.Regu = 1; 
% [eigvector, eigvalue] = LGE(data, Sp, Sc, options); 
% 
%========================== 
if (~isfield(options,'Regu') | ~options.Regu)  
    if isfield(options,'Fisherface') & options.Fisherface 
        options.PCARatio = nSmp - nLabel; 
    else 
        error('PCARatio is not correct!'); 
    end 
end 
 
 
if bGlobal & isfield(options,'keepMean') & options.keepMean 
    [eigvector, eigvalue, elapse] = LGE(Sp, Sc, options); 
else 
    [eigvector, eigvalue, elapse] = LGE(Sp, Sc, options, data); 
end 
 
elapse.timeW = timeW; 
elapse.timeAll = elapse.timeAll + elapse.timeW; 
 
 
eigIdx = find(eigvalue < 1e-10); 
eigvalue (eigIdx) = []; 
eigvector(:,eigIdx) = []; 
 
 
 
 
 

 
