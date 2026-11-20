function F = mpm_gb_predict(mdl, X, nTrees)
%MPM_GB_PREDICT Raw score of a boosted tree ensemble (MPM_XGBOOST_TRAIN / MPM_LIGHTGBM_TRAIN).
%
%   F = mpm_gb_predict(mdl, X)          all trees;  F = base + sum_t f_t(x)
%   F = mpm_gb_predict(mdl, X, nTrees)  first nTrees trees only
%   Apply MPM_GB_LOSS('link', ...) for probabilities. Trees are traversed for all rows at once
%   (one vectorised step per tree level).
X = double(X);
n = size(X, 1);
if nargin < 3, nTrees = numel(mdl.trees); end
F = mdl.base * ones(n, 1);
for t = 1:min(nTrees, numel(mdl.trees))
    F = F + tree_value(mdl.trees{t}, X);
end
end

function v = tree_value(tr, X)
n = size(X, 1);
node = ones(n, 1);
inner = tr.feat(node) > 0;
while any(inner)
    ii = find(inner);
    nd = node(ii);
    x = X(sub2ind(size(X), ii, tr.feat(nd)));
    goLeft = x <= tr.thr(nd);
    nd(goLeft) = tr.left(nd(goLeft));
    nd(~goLeft) = tr.right(nd(~goLeft));
    node(ii) = nd;
    inner(ii) = tr.feat(nd) > 0;
end
v = tr.value(node);
v = v(:);
end
