function Dict=JointDictLearning(X, W,par);
%% function: To learn the dictionary Df, Dw and the mapping matrix M
%% input:
%   X : D x N matrix
%   W : D x N matrix
%  par : parameters for dictionary learning
%% output: Dict
%     Dict.Dx  -- the dictionary of X space   
%     Dict.Dw  -- the dictionary of W space
%     Dict.Mf  -- Mapping Alphaw = Mf * Alphaf;
%     Dict.Mw  -- Mapping Alphaf = Mw * Alphaw;
%     Dict.f   -- objetive values
%% 
fprintf('Least Square Semi-Coupled Dictionary Learning...');   
Dx =rand(size(X,1),par.K);
Dw =rand(size(W,1),par.K);
Mx = eye(size(Dw, 2));
Mw = eye(size(Dx, 2));
Alphax = rand(par.K,size(X,2));
Alphaw = rand(par.K,size(W,2));
[Alphax, Alphaw, X, W, Dx, Dw, Mx, Mw, f] = LSSCDL(Alphax, Alphaw,X, W, Dx, Dw, Mx, Mw, par);
Dict.Dx = Dx;    
Dict.Dw = Dw;
Dict.Mx = Mx;
Dict.Mw= Mw;
Dict.f = f;
fprintf('Done!\n');
end