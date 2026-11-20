function fold = mpm_kfold(n, k, rs, y, groups)
%MPM_KFOLD Fold labels (1..k) for cross-validation.
%
%   fold = mpm_kfold(n, k, rs)               plain random folds
%   fold = mpm_kfold(n, k, rs, y)            stratified by class (logical / categorical y)
%   fold = mpm_kfold(n, k, rs, y, groups)    grouped: all samples of a group share a fold
%                                            (e.g. the buffered cells of one deposit), stratified
%                                            by the class of the group
if nargin < 4 || isempty(y), y = ones(n, 1); end
if nargin < 5 || isempty(groups), groups = (1:n)'; end
y = y(:); groups = groups(:);
[ug, first, g] = unique(groups, 'stable');
gy = y(first);
fold = zeros(n, 1);
gf = zeros(numel(ug), 1);
for c = unique(gy)'
    ii = find(gy == c);
    ii = ii(randperm(rs, numel(ii)));
    off = randi(rs, k) - 1;                             % do not always start at fold 1
    gf(ii) = mod((0:numel(ii)-1)' + off, k) + 1;
end
fold(:) = gf(g);
end
