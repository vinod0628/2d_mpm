function C = mpm_eval_map(name, allScores, posScores, negScores, nThresholds)
%MPM_EVAL_MAP Map-based validation of a prospectivity score (same convention as the Python
%   evaluation/metrics.py): positives = deposit cells, negatives = every other valid cell.
%
%   C = mpm_eval_map(name, allScores, posScores, negScores)
%
%   C fields
%     roc                       ROC of deposit cells vs. all background cells (MPM_ROC)
%     auc
%     area_frac, capture_frac   success-rate (training) / prediction-rate (test) curve:
%                               fraction of study area (highest scores first) vs. fraction of
%                               deposits captured; sr_auc = area under it
%     pa_thr, pa_pred, pa_area  prediction-area (P-A) plot: % deposits and % area with score >= thr
%     pa_intersection           struct(threshold, prediction, area) where prediction = 100 - area
if nargin < 5, nThresholds = 400; end
allScores = double(allScores(:)); allScores = allScores(isfinite(allScores));
posScores = double(posScores(:)); negScores = double(negScores(:));
C.name = name;
C.npos = numel(posScores); C.nneg = numel(negScores);
C.roc = mpm_roc([true(C.npos, 1); false(C.nneg, 1)], [posScores; negScores]);
C.auc = C.roc.auc;

qs = quantile(allScores, linspace(0, 1, nThresholds))';
thr = unique([qs; posScores]);                          % ascending
areaGe = count_ge(allScores, thr) / numel(allScores);
capGe = count_ge(posScores, thr) / max(numel(posScores), 1);
% success-rate: highest thresholds first
C.area_frac = [0; flipud(areaGe); 1];
C.capture_frac = [0; flipud(capGe); 1];
C.sr_auc = trapz(C.area_frac, C.capture_frac);

C.pa_thr = thr; C.pa_pred = 100 * capGe; C.pa_area = 100 * areaGe;
C.pa_intersection = struct('threshold', NaN, 'prediction', NaN, 'area', NaN);
d = C.pa_pred - (100 - C.pa_area);
i = find(diff(sign(d)) ~= 0, 1);
if ~isempty(i)
    f = 0; if d(i) ~= d(i+1), f = d(i) / (d(i) - d(i+1)); end
    lerp = @(v) v(i) + f * (v(i+1) - v(i));
    C.pa_intersection = struct('threshold', lerp(thr), 'prediction', lerp(C.pa_pred), 'area', lerp(C.pa_area));
end
end

function c = count_ge(values, thr)
% number of VALUES >= each threshold (thr ascending, unique)
counts = histcounts(values, [thr; Inf]);
c = flipud(cumsum(flipud(counts(:))));
end
