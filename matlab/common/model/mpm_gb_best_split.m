function s = mpm_gb_best_split(G, H, C, nbins, o, allowed)
%MPM_GB_BEST_SPLIT Best split of one tree node from per-feature gradient/hessian histograms.
%
%   s = mpm_gb_best_split(G, H, C, nbins, o, allowed)
%
%   G, H, C  nb x P sums of gradients, hessians and sample counts per (bin, feature) of the node
%   nbins    1 x P number of bins per feature;  allowed  1 x P logical (column subsampling)
%   o.lambda (L2 on leaf weights), o.alpha (L1), o.min_hess (min hessian sum per child =
%   XGBoost min_child_weight / LightGBM min_sum_hessian_in_leaf), o.min_count (min samples per child)
%
%   Second-order split gain (Chen & Guestrin 2016, Eq. 7; the same score is used by LightGBM):
%       gain = 1/2 [ S(GL)^2/(HL+lambda) + S(GR)^2/(HR+lambda) - S(G)^2/(H+lambda) ]
%   with S(.) the L1 soft-threshold (S(g) = g when alpha = 0). The complexity penalty gamma /
%   min_gain_to_split is applied by the caller.
%   s fields: gain (-Inf if no valid split), feature, bin (split after this bin), GL, HL, GR, HR
GL = cumsum(G, 1); HL = cumsum(H, 1); CL = cumsum(C, 1);
Gt = GL(end, :); Ht = HL(end, :); Ct = CL(end, :);
GR = Gt - GL; HR = Ht - HL; CR = Ct - CL;
score = @(g, h) soft(g, o.alpha).^2 ./ (h + o.lambda);
gain = 0.5 * (score(GL, HL) + score(GR, HR) - score(Gt, Ht));
nb = size(G, 1);
valid = (1:nb)' < nbins(:)' & HL >= o.min_hess & HR >= o.min_hess & CL >= o.min_count & CR >= o.min_count;
if nargin > 5 && ~isempty(allowed), valid = valid & allowed(:)'; end
gain(~valid) = -Inf;
[best, k] = max(gain(:));
s = struct('gain', best, 'feature', 0, 'bin', 0, 'GL', 0, 'HL', 0, 'GR', 0, 'HR', 0);
if isfinite(best)
    [s.bin, s.feature] = ind2sub(size(gain), k);
    s.GL = GL(k); s.HL = HL(k); s.GR = GR(k); s.HR = HR(k);
end
end

function v = soft(g, a)
if a > 0, v = sign(g) .* max(abs(g) - a, 0); else, v = g; end
end
