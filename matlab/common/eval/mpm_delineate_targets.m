function [classMap, T, thr] = mpm_delineate_targets(scoreMap, valid, trainLin, testLin, opts)
%MPM_DELINEATE_TARGETS Classify a prospectivity map into exploration-target classes.
%
%   [classMap, T, thr] = mpm_delineate_targets(scoreMap, valid, trainLin, testLin, opts)
%
%   opts.method 'area' (default): classes by occupied area, opts.levels = [0.05 0.10 0.20] gives
%                 4 = top 5 % of the study area, 3 = 5-10 %, 2 = 10-20 %, 1 = remaining area
%               'threshold': opts.levels = score thresholds (ascending), e.g. the P-A intersection
%   classMap  uint8 (0 = outside study area, 1 = lowest ... K = highest class)
%   T         table: class, label, min_score, area_pct, train_deposits_pct, test_deposits_pct,
%             train/test deposit density ratio (deposit % / area %)
if nargin < 5, opts = struct(); end
opts = mpm_merge(struct('method', 'area', 'levels', [0.05 0.10 0.20]), opts);
s = double(scoreMap(valid));
switch lower(opts.method)
    case 'area'
        lv = sort(opts.levels(:), 'descend');            % 0.20 0.10 0.05
        thr = quantile(s, 1 - lv);                       % ascending thresholds
        labels = [sprintf("remaining %g %%", 100 * (1 - max(lv))); ...
            string(arrayfun(@(a, b) sprintf("top %g-%g %%", 100 * b, 100 * a), lv(1:end-1), lv(2:end), 'UniformOutput', false)); ...
            sprintf("top %g %%", 100 * min(lv))];
    case 'threshold'
        thr = sort(opts.levels(:));
        labels = [sprintf("< %.3g", thr(1)); string(arrayfun(@(a, b) sprintf("%.3g - %.3g", a, b), thr(1:end-1), thr(2:end), 'UniformOutput', false)); ...
            sprintf(">= %.3g", thr(end))];
    otherwise
        error('mpm:targets', 'Unknown target method "%s".', opts.method);
end
thr = thr(:);
cls = ones(size(s));
for k = 1:numel(thr), cls(s >= thr(k)) = k + 1; end
classMap = zeros(size(valid), 'uint8');
classMap(valid) = uint8(cls);

K = numel(thr) + 1;
pctArea = zeros(K, 1); pTr = zeros(K, 1); pTe = zeros(K, 1);
cTr = double(classMap(trainLin)); cTe = double(classMap(testLin));
for k = 1:K
    pctArea(k) = 100 * mean(cls == k);
    pTr(k) = 100 * mean(cTr == k);
    pTe(k) = 100 * mean(cTe == k);
end
if isempty(testLin), pTe(:) = NaN; end
minScore = [min(s); thr];
T = table((1:K)', labels, minScore, pctArea, pTr, pTe, pTr ./ pctArea, pTe ./ pctArea, ...
    'VariableNames', {'class', 'label', 'min_score', 'area_pct', 'train_deposits_pct', ...
    'test_deposits_pct', 'train_density_ratio', 'test_density_ratio'});
T = flipud(T);
end
