function mdl = mpm_iforest_train(X, o)
%MPM_IFOREST_TRAIN Isolation forest (Liu, Ting & Zhou 2008), written from the paper.
%
%   mdl = mpm_iforest_train(X, opts)      X: N x P, rows = (unlabelled) observations
%   opts (all optional):
%     n_trees       t, number of isolation trees                                (100)
%     sample_size   psi, sub-sampling size per tree, drawn WITHOUT replacement   (256)
%     height_limit  'auto' = ceiling(log2 psi) (paper) | 'full' (grow until isolated) | number
%
%   Algorithm 1  iForest(X, t, psi): for each tree draw psi points without replacement and build
%                iTree(X', 0, l) with height limit l = ceiling(log2 psi).
%   Algorithm 2  iTree(X, e, l): if e >= l or |X| <= 1 -> external node {Size = |X|}; else pick a
%                random attribute q and a random split value p uniformly between min and max of
%                q in X; X_l = {x: x_q < p}, X_r = {x: x_q >= p}.
%                Implementation detail not specified in the paper: q is drawn among the
%                attributes that are NOT constant in X (a constant attribute cannot split);
%                if every attribute is constant the node becomes external.
%   Algorithm 3  PathLength(x, T, e): edges e from the root to the external node reached by x,
%                plus c(Size) for the unbuilt subtree of an external node with Size > 1.
%   Score (Eq. 2)  s(x, psi) = 2^(-E(h(x)) / c(psi)),  c(n) = 2H(n-1) - 2(n-1)/n  (Eq. 1),
%                H(i) = harmonic number = psi(i+1) + Euler gamma (exact, via the digamma fn).
%                s -> 1: anomaly; s << 0.5: normal; all s ~ 0.5: no distinct anomaly.
%   Scoring: MPM_IFOREST_SCORE.
%
%   Reference: Liu, F.T., Ting, K.M., Zhou, Z.-H., 2008. Isolation forest. Proc. 8th IEEE Int.
%   Conf. on Data Mining (ICDM), 413-422.
if nargin < 2, o = struct(); end
o = mpm_merge(struct('n_trees', 100, 'sample_size', 256, 'height_limit', 'auto'), o);
X = double(X);
[N, P] = size(X);
psi = min(round(o.sample_size), N);
if psi < 2, error('mpm:iforest', 'Need at least 2 observations.'); end
if ischar(o.height_limit) || isstring(o.height_limit)
    switch lower(o.height_limit)
        case 'auto', L = ceil(log2(psi));
        case 'full', L = psi - 1;                    % a tree on psi points never gets deeper
        otherwise, error('mpm:iforest', 'height_limit: ''auto'' | ''full'' | number.');
    end
else
    L = round(o.height_limit);
end
mdl.trees = cell(o.n_trees, 1);
for t = 1:o.n_trees
    Xs = X(randperm(N, psi), :);                    % sub-sample without replacement
    mdl.trees{t} = itree(Xs, L, P);
end
mdl.n_trees = o.n_trees; mdl.psi = psi; mdl.height_limit = L;
mdl.c_psi = mpm_iforest_c(psi);
mdl.mean_depth = mean(cellfun(@(T) max(T.depth), mdl.trees));
end

function T = itree(X, L, P)
% iterative version of Algorithm 2 (explicit stack, nodes numbered in creation order)
n = size(X, 1); maxNodes = 2 * n - 1;
att = zeros(maxNodes, 1); val = zeros(maxNodes, 1);
left = zeros(maxNodes, 1); right = zeros(maxNodes, 1);
sz = zeros(maxNodes, 1); depth = zeros(maxNodes, 1);
rows = cell(maxNodes, 1); rows{1} = (1:n)';
nn = 1; stack = 1;
while ~isempty(stack)
    k = stack(end); stack(end) = [];
    r = rows{k}; rows{k} = [];
    sz(k) = numel(r);
    if depth(k) >= L || numel(r) <= 1, continue; end          % external node
    Xr = X(r, :);
    lo = min(Xr, [], 1); hi = max(Xr, [], 1);
    q = find(hi > lo);
    if isempty(q), continue; end                               % all attributes constant
    q = q(randi(numel(q)));
    p = lo(q) + rand * (hi(q) - lo(q));
    if p <= lo(q), p = hi(q); end                              % rand == 0 -> keep both sides non-empty
    isL = Xr(:, q) < p;
    att(k) = q; val(k) = p;
    left(k) = nn + 1; right(k) = nn + 2;
    rows{nn + 1} = r(isL); rows{nn + 2} = r(~isL);
    depth(nn + 1:nn + 2) = depth(k) + 1;
    stack = [stack, nn + 2, nn + 1]; %#ok<AGROW>
    nn = nn + 2;
end
T.att = att(1:nn); T.val = val(1:nn); T.left = left(1:nn); T.right = right(1:nn);
T.size = sz(1:nn); T.depth = depth(1:nn);
T.leaf_c = zeros(nn, 1);
ext = T.att == 0;
T.leaf_c(ext) = T.depth(ext) + mpm_iforest_c(T.size(ext));   % h(x) of a point ending in this leaf
T.heap = to_heap(T, L);
end

function H = to_heap(T, L)
% Complete-binary-heap layout of a height-limited tree (children of heap slot j: 2j, 2j+1).
% An external node at depth e < L continues as a chain of always-left dummy slots (attribute 1,
% split +Inf), so every point reaches level L after exactly L steps; its h(x) is stored there.
if L > 20, H = []; return; end                         % 'full' trees: too deep for a heap
nh = 2 ^ (L + 1) - 1;
H.att = ones(nh, 1); H.val = inf(nh, 1); H.leafc = zeros(nh, 1); H.L = L;
slot = zeros(numel(T.att), 1); slot(1) = 1;
for k = 1:numel(T.att)                                 % parents are created before children
    j = slot(k);
    if T.att(k) > 0
        H.att(j) = T.att(k); H.val(j) = T.val(k);
        slot(T.left(k)) = 2 * j; slot(T.right(k)) = 2 * j + 1;
    else
        H.leafc(j * 2 ^ (L - T.depth(k))) = T.leaf_c(k);   % end of the always-left chain
    end
end
end
