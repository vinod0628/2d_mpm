function Phi = mpm_rvm_basis(X, C, kernel, s, bias)
%MPM_RVM_BASIS Design matrix of an RVM: kernel between samples X (N x P) and centres C (M x P).
%   'gaussian': k(x, c) = exp(-||x - c||^2 / s^2)      'linear': k(x, c) = x' * c
%   bias = true prepends a column of ones.
switch lower(kernel)
    case {'gaussian', 'rbf'}
        Phi = exp(-pdist2(double(X), double(C), 'squaredeuclidean') / s^2);
    case 'linear'
        Phi = double(X) * double(C)';
    otherwise
        error('mpm:rvm', 'Unknown RVM kernel "%s".', kernel);
end
if bias, Phi = [ones(size(X, 1), 1), Phi]; end
end
