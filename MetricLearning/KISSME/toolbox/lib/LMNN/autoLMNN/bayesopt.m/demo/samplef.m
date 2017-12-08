% function [Lval,Cval] = samplef(X,b)
% Sample function for bayesopt.m 
% To use bayesopt.m you need an opt struct (see demo.m or the readme) and a function handle to a function like this.
% The function should return two arguments, the objective function value and the constraint function value.
%
% The function handle should ultimately have only one argument, a vector of parameters X.
% The function itself can have additional parameters that are passed in as constants. For example:
%          b = 1;
%          F = @(X)samplef(X,b);
% 
% This lets you, for example, pass in datasets when tuning ML algorithm parameters.
function [Lval,Cval] = samplef2(x,y,b)
	L = @(x,y) cos(2.*x).*cos(y) + sin(b.*x);
	C = @(x,y) -(-cos(x).*cos(y)+sin(x).*sin(y));
	Lval = L(x,y) + 1e-4*randn(1,size(x,2));
    Cval = C(x,y) + 1e-4*randn(1,size(y,2));
end
