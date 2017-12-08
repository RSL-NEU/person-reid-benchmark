%% This function tests the SPAMS installation.
function r = testSPAMS()
r = true;
try
    X = randn(100,1000); D = randn(100,50);
    param.lambda=0.15; 
    param.mode=2;
    alpha = mexLasso(X,D,param);
catch
   r = false;
end

return
