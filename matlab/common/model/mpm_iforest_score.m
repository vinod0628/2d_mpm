function [s, Eh] = mpm_iforest_score(mdl, X)
%MPM_IFOREST_SCORE Anomaly score of an isolation forest (Liu et al. 2008, Algorithm 3 + Eq. 2).
%
%   [s, Eh] = mpm_iforest_score(mdl, X)
%   s   N x 1 anomaly score  2^(-E(h(x)) / c(psi))  in (0, 1]
%   Eh  N x 1 mean path length E(h(x)) over the trees
%
%   Vectorised PathLength: every point descends one level per step.
%   - height-limited trees: complete-heap layout (children of slot j = 2j, 2j+1; external nodes
%     continue as always-left dummy slots), exactly L steps, 2 table look-ups per step;
%   - 'full' trees: node tables, external nodes point to themselves.
%   Both are identical to the recursive PathLength of the paper (checked).
X = double(X); n = size(X, 1); P = size(X, 2);
Xt = X';                                  % P x N: a point's attributes are contiguous
base = (0:n - 1)' * P;
Eh = zeros(n, 1);
for t = 1:mdl.n_trees
    T = mdl.trees{t};
    if isfield(T, 'heap') && ~isempty(T.heap)
        H = T.heap; k = ones(n, 1);
        for d = 1:H.L
            k = 2 * k + (Xt(base + H.att(k)) >= H.val(k));
        end
        Eh = Eh + H.leafc(k);
    else
        ext = T.att == 0; self = find(ext);
        att = T.att; att(ext) = 1; val = T.val; val(ext) = Inf;
        lft = T.left; lft(ext) = self; rgt = T.right; rgt(ext) = self;
        k = ones(n, 1);
        for d = 1:max(T.depth)
            goL = Xt(base + att(k)) < val(k);
            k = rgt(k) + goL .* (lft(k) - rgt(k));
        end
        Eh = Eh + T.leaf_c(k);
    end
end
Eh = Eh / mdl.n_trees;
s = 2 .^ (-Eh / mdl.c_psi);
end
