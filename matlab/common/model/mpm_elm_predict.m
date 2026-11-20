function Y = mpm_elm_predict(mdl, X, block)
%MPM_ELM_PREDICT Output of a trained ELM,  f(x) = h(x) beta  (linear output nodes).
%
%   Y = mpm_elm_predict(mdl, X)          mdl from MPM_ELM_TRAIN
%   Y = mpm_elm_predict(mdl, X, block)   rows per block (default 20000) - bounds the N x L
%                                        hidden matrix when predicting whole maps
if nargin < 3 || isempty(block), block = 20000; end
n = size(X, 1);
Y = zeros(n, size(mdl.beta, 2));
for s = 1:block:n
    i = s:min(s + block - 1, n);
    Y(i, :) = mpm_elm_hidden(mdl, X(i, :)) * mdl.beta;
end
end
