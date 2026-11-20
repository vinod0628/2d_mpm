function [f, sd, F] = mpm_pubag_predict(mdl, X, block)
%MPM_PUBAG_PREDICT Score of a PU-bagging model (MPM_PUBAG_TRAIN), Algorithm 1 of Mordelet & Vert (2014).
%
%   [f, sd, F] = mpm_pubag_predict(mdl, X)
%   f   N x 1 aggregated score  1/T sum_t f_t(x)   (svm: decision values; tree: P(class 1) of
%       each balanced P-vs-U_t tree)
%   sd  N x 1 standard deviation of the member scores (bagging spread)
%   F   N x T member scores
%   block = rows per block (default 20000) bounds the memory of the kernel matrix.
if nargin < 3 || isempty(block), block = 20000; end
X = double(X); n = size(X, 1); T = mdl.n_bags;
f = zeros(n, 1); sd = zeros(n, 1);
keepF = nargout > 2; if keepF, F = zeros(n, T); end
for i = 1:block:n
    j = i:min(i + block - 1, n);
    switch mdl.base
        case 'svm'      % all members share the kernel: one kernel matrix, one product per block
            Fj = exp(-pdist2(X(j, :), mdl.Z, 'squaredeuclidean') / mdl.kernel_scale ^ 2) * mdl.Coef + mdl.bias;
        case 'tree'
            Fj = zeros(numel(j), T);
            for t = 1:T
                [~, sc] = predict(mdl.trees{t}, X(j, :));
                Fj(:, t) = sc(:, 2);
            end
    end
    f(j) = mean(Fj, 2);
    if nargout > 1, sd(j) = std(Fj, 0, 2); end
    if keepF, F(j, :) = Fj; end
end
end
