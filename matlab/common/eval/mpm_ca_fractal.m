function CA = mpm_ca_fractal(values, opts)
%MPM_CA_FRACTAL Concentration-area (C-A) fractal model with automatic break detection.
%
%   CA = mpm_ca_fractal(values, opts)
%
%   values  scores of every valid cell (prospectivity, grade, element concentration, ...);
%           only positive values enter the log-log plot
%   opts.n_thresholds  number of log-spaced thresholds (default 100)
%   opts.segments      number of straight-line segments (populations) fitted to the
%                      log10(area >= t) vs log10(t) curve, default 3
%   opts.min_points    minimum thresholds per segment (default 6)
%   opts.cell_area     area of one cell (e.g. km^2; default 1 -> area in cells)
%   opts.top_exclude   fraction of the highest values left out of the fit (default 0.001): near the
%                      maximum only a handful of cells remain and log(area) drops steeply - an end
%                      effect that would otherwise be fitted as a spurious "anomaly" segment
%
%   Model (Cheng, Agterberg & Ballantyne 1994): A(rho >= t) ~ t^(-D). Different populations
%   (background, intermediate, anomaly / target) plot as straight lines with different slopes
%   on log-log axes; the breaks between the lines are the thresholds. Segments are fitted by
%   least squares; the break positions minimise the total squared error (exact dynamic
%   programming over all segmentations, so the result is reproducible - no manual picking).
%
%   CA fields
%     t, area, logt, logA      the C-A curve (thresholds ascending)
%     breaks                   threshold values at the segment boundaries (ascending)
%     segments                 table: segment, t_from, t_to, slope, intercept, D (= -slope), r2,
%                              n_points, area_pct (area with t >= t_from)
%     fit_logA                 fitted log10 area at each threshold (for plotting)
%
%   Reference: Cheng, Q., Agterberg, F.P., Ballantyne, S.B., 1994. The separation of
%   geochemical anomalies from background by fractal methods. J. Geochem. Explor. 51, 109-130.
if nargin < 2, opts = struct(); end
opts = mpm_merge(struct('n_thresholds', 100, 'segments', 3, 'min_points', 6, 'cell_area', 1, ...
    'top_exclude', 0.001), opts);
v = double(values(:)); v = v(isfinite(v));
nAll = numel(v);
p = sort(v(v > 0));
if numel(p) < 10, error('mpm:ca', 'C-A fractal needs positive values (found %d).', numel(p)); end
lo = p(max(1, ceil(0.001 * numel(p))));        % ignore the extreme low tail (log of ~0)
hi = p(max(1, min(numel(p), floor((1 - opts.top_exclude) * numel(p)))));   % end effect
t = unique(logspace(log10(lo), log10(hi), opts.n_thresholds)');
cnt = numel(p) - (arrayfun(@(x) find(p >= x, 1), t) - 1);      % cells with value >= t
keep = cnt > 0;
t = t(keep); cnt = cnt(keep);
CA.t = t; CA.area = cnt * opts.cell_area;
CA.logt = log10(t); CA.logA = log10(CA.area);
CA.n_cells = nAll; CA.cell_area = opts.cell_area;

% ---------------------------------------------------------- optimal segmentation (DP)
x = CA.logt; y = CA.logA; n = numel(x);
K = min(opts.segments, floor(n / opts.min_points));
K = max(K, 1);
Sx = [0; cumsum(x)]; Sy = [0; cumsum(y)]; Sxx = [0; cumsum(x.^2)]; Sxy = [0; cumsum(x.*y)]; Syy = [0; cumsum(y.^2)];
sse = @(i, j) seg_sse(i, j, Sx, Sy, Sxx, Sxy, Syy);
m = opts.min_points;
cost = inf(K, n); arg = zeros(K, n);
for j = m:n, cost(1, j) = sse(1, j); end
for k = 2:K
    for j = k*m:n
        best = inf; bi = 0;
        for i = (k-1)*m+1 : j-m+1                 % segment k = i..j
            c = cost(k-1, i-1) + sse(i, j);
            if c < best, best = c; bi = i; end
        end
        cost(k, j) = best; arg(k, j) = bi;
    end
end
starts = zeros(K, 1); j = n;
for k = K:-1:2, starts(k) = arg(k, j); j = starts(k) - 1; end
starts(1) = 1;
ends = [starts(2:end) - 1; n];

rows = cell(K, 9); CA.fit_logA = nan(n, 1);
for k = 1:K
    i = starts(k):ends(k);
    b = polyfit(x(i), y(i), 1);
    f = polyval(b, x(i)); CA.fit_logA(i) = f;
    r2 = 1 - sum((y(i) - f).^2) / max(sum((y(i) - mean(y(i))).^2), eps);
    rows(k, :) = {k, t(i(1)), t(i(end)), b(1), b(2), -b(1), r2, numel(i), 100 * cnt(i(1)) / nAll};
end
CA.segments = cell2table(rows, 'VariableNames', {'segment', 't_from', 't_to', 'slope', 'intercept', ...
    'D', 'r2', 'n_points', 'area_pct'});
CA.breaks = t(starts(2:end));
end

function s = seg_sse(i, j, Sx, Sy, Sxx, Sxy, Syy)
% residual sum of squares of the least-squares line through points i..j
n = j - i + 1;
sx = Sx(j+1) - Sx(i); sy = Sy(j+1) - Sy(i);
sxx = Sxx(j+1) - Sxx(i); sxy = Sxy(j+1) - Sxy(i); syy = Syy(j+1) - Syy(i);
vx = sxx - sx^2 / n; vy = syy - sy^2 / n; cxy = sxy - sx * sy / n;
if vx <= eps, s = max(vy, 0); else, s = max(vy - cxy^2 / vx, 0); end
end
