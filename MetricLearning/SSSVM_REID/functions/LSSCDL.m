function [Alphax, Alphaw, X, W, Dx, Dw, Mx, Mw,f] = LSSCDL(Alphax, Alphaw, X, W, Dx, Dw, Mx, Mw, par)
%% function: Least Square Semi-Coupled Dictionary Learning
%% input:
%   X : D x N matrix
%   W : D x N matrix
%   Alphax, Alphaw, Dx, Dw, Mx, Mw : initialized matrices
%   par : parameters for dictionary learning
%% output: 
%     Dx  -- the dictionary of X space
%     Dw  -- the dictionary of W space
%     Mx  -- Mapping Alphaw = Mx * Alphax;
%     Mw  -- Mapping Alphaf = Mw * Alphaw;
%     f   -- objetive values
%% 

lambda1  =  par.lambda1;
lambdac  =  par.lambdac;
lambdam  =  par.lambdam;
lambdad  =  par.lambdad;
nIter    =  par.nIter;
epsilon  =  par.epsilon;

f = 0;
for t = 1 :  nIter
    f_prev = f;
    Alphax = inv(Dx'*Dx + lambda1* Mx'* Mx + (lambda1+lambdac) * eye(size(Alphax, 1)))*(Dx'*X+lambda1*(Mx'+Mw)*Alphaw);
    Alphaw = inv(Dw'*Dw + lambda1* Mw'* Mw + (lambda1+lambdac) * eye(size(Alphaw, 1)))*(Dw'*W+lambda1*(Mw'+Mx)*Alphax);
%     Alphax = inv(Dx'*Dx + lambda1* Mx'* Mx + lambdac * eye(size(Alphax, 1)))*(Dx'*X+lambda1*Mx'*Alphaw);
%     Alphaw = inv(Dw'*Dw + (lambda1+lambdac) * eye(size(Alphax, 1)))*(Dw'*W+lambda1*Mx*Alphax);
    % Update D
    Dw=W * Alphaw' * inv(Alphaw * Alphaw' + lambdad * eye(size(Alphaw, 1))) ;
    Dx=X * Alphax' * inv(Alphax * Alphax' + lambdad * eye(size(Alphax, 1))) ;
    % Update M
    Mw =  Alphax * Alphaw' * inv(Alphaw * Alphaw' + (lambdam./lambda1) * eye(size(Alphaw, 1))) ;
    Mx =  Alphaw * Alphax' * inv(Alphax * Alphax' + (lambdam./lambda1) * eye(size(Alphax, 1))) ;     
    % objective function value
    fp=norm(X - Dx * Alphax,'fro')+ norm(W - Dw * Alphaw,'fro')+ lambda1 * norm( Alphaw - Mx * Alphax,'fro')+lambda1 * norm( Alphax - Mw * Alphaw,'fro');
    fs= lambdac * (norm(Alphax,'fro')+norm(Alphaw,'fro'))+ lambdam * norm(Mx, 'fro')+lambdam * norm(Mw, 'fro')+lambdad *(norm(Dx,'fro')+norm(Dw,'fro'));
    f = fp + fs;
    if (abs(f_prev - f) / f < epsilon)
        break;
    end
%     fprintf('Iter %d: Objective Value: %f \n', t,f);
end
end