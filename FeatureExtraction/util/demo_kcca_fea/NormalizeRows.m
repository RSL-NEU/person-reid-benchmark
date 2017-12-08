function [b, normF] = NormalizeRows(a, type)
% Normalizes the rows of a using specified normalisation type. Makes sure 
% there is no division by zero: b will not contain any NaN entries.
%
% a:            data with row vectors
% type:         String:
%                   'L1' (default) Sums to 1
%                   'L2' Unit Length
%                   'Sqrt->L1' Square root followed by L1
%                   'L1->Sqrt' L1 followed by Square root. Becomes unit length
%                   'Sqrt->L2' Square root followed by L2. Square rooting
%                               while keeping the sign
%                   'None' No normalization
% 
% b:            normalized data with row vecors. 
% normF:        Normalization factor per row (when valid)
%
% Jasper Uijlings, 2011

% Default: L1
if nargin == 1
    type = 'L1';
end

switch type
    case 'L1'
        normF = sum(a,2); % Get sums
        normFUsed = normF;
        normFUsed(normFUsed == 0) = 1; % Prevent division by zero
        b = bsxfun(@rdivide, a, normFUsed); % Normalise
    case 'L2'
        normF = sqrt(sum(a .* a, 2)); % Get length
        normFUsed = normF;
        normFUsed(normFUsed == 0) = 1; % Prevent division by zero
        b = bsxfun(@rdivide, a, normFUsed);
    case 'Sqrt->L1'
        b = NormalizeRows(sqrt(a), 'L1');
        normF = [];
    case 'L1->Sqrt' % This is the same as ROOTSIFT
        b = sqrt(NormalizeRows(a, 'L1'));
        normF = [];
    case 'Sqrt->L2'
        b = NormalizeRows(SquareRootAbs(a), 'L2');
        normF = [];
    case 'None'
        b = a;
        normF = [];
    otherwise 
        error('Wrong normalization type given');
end
