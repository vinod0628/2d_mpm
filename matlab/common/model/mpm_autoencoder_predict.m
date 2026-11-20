function [err, Xh] = mpm_autoencoder_predict(mdl, X, block)
%MPM_AUTOENCODER_PREDICT Reconstruction and reconstruction error of an autoencoder.
%
%   [err, Xh] = mpm_autoencoder_predict(mdl, X)   err = mean squared error per row, ||x_hat - x||^2 / P
%   block = rows per block (default 50000) bounds memory for whole-map prediction.
if nargin < 3 || isempty(block), block = 50000; end
X = double(X); n = size(X, 1);
err = zeros(n, 1);
if nargout > 1, Xh = zeros(size(X)); end
for s = 1:block:n
    i = s:min(s + block - 1, n);
    A = X(i, :);
    for l = 1:numel(mdl.W)
        Z = A * mdl.W{l} + mdl.b{l};
        switch mdl.act{l}
            case 'linear',  A = Z;
            case 'tanh',    A = tanh(Z);
            case 'relu',    A = max(Z, 0);
            case 'sigmoid', A = 1 ./ (1 + exp(-Z));
        end
    end
    err(i) = mean((A - X(i, :)) .^ 2, 2);
    if nargout > 1, Xh(i, :) = A; end
end
end
