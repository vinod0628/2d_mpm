function [Bin, cuts, nbins] = mpm_gb_binning(X, maxBin)
%MPM_GB_BINNING Discretise features into bins for gradient-boosted tree split finding.
%
%   [Bin, cuts, nbins] = mpm_gb_binning(X, maxBin)
%
%   maxBin = Inf  -> one bin per distinct value: thresholds at every midpoint between consecutive
%                    distinct values = the EXACT greedy split enumeration of XGBoost (Chen &
%                    Guestrin 2016, Alg. 1)
%   maxBin = k    -> at most k equal-frequency (quantile) bins = histogram / approximate split
%                    finding (XGBoost Alg. 2 with uniform weights; LightGBM histogram algorithm,
%                    Ke et al. 2017, Alg. 1)
%
%   cuts{j}  ascending thresholds of feature j; a value x falls in bin k if cuts(k-1) < x <= cuts(k)
%            (bin 1: x <= cuts(1); last bin: x > cuts(end)). A split "after bin k" therefore sends
%            x <= cuts{j}(k) to the left child - the rule used for prediction.
%   Bin      n x P uint16 bin indices (1-based); nbins(j) = numel(cuts{j}) + 1
X = double(X);
[n, P] = size(X);
Bin = zeros(n, P, 'uint16');
cuts = cell(1, P);
nbins = ones(1, P);
for j = 1:P
    x = X(:, j);
    u = unique(x);
    if numel(u) <= maxBin
        c = (u(1:end-1) + u(2:end)) / 2;
    else
        % equal-frequency bin boundaries, placed midway between distinct values
        r = round((1:maxBin-1) / maxBin * numel(x));
        xs = sort(x);
        lo = xs(max(r, 1)); c = zeros(numel(lo), 1);
        for k = 1:numel(lo)
            nxt = u(find(u > lo(k), 1));
            if isempty(nxt), c(k) = NaN; else, c(k) = (lo(k) + nxt) / 2; end
        end
        c = unique(c(~isnan(c)));
    end
    if numel(c) >= intmax('uint16'), error('mpm:gb', 'Too many bins for feature %d.', j); end
    cuts{j} = c(:);
    nbins(j) = numel(c) + 1;
    if isempty(c)
        Bin(:, j) = 1;
    else
        Bin(:, j) = uint16(discretize(x, [-Inf; c(:); Inf], 'IncludedEdge', 'right'));
    end
end
end
