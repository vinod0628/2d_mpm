function B = mpm_binary_targets(scoreMap, valid, trainLin, testLin, opts)
%MPM_BINARY_TARGETS Binary target map (1 = predicted deposit area, 0 = non-deposit) from a
%   prospectivity map, thresholded by the P-A intersection and/or the C-A fractal.
%
%   B = mpm_binary_targets(scoreMap, valid, trainLin, testLin, opts)
%
%   scoreMap   H x W prospectivity (any "higher = more prospective" score)
%   valid      H x W logical study-area mask
%   trainLin / testLin  linear cell indices of training / test deposits ([] if none)
%   opts.methods     {'pa', 'ca'} (default both); 'pa' needs deposits
%   opts.pa_deposits 'train' (default) | 'all' - deposits defining the P-A curve
%   opts.ca          options for MPM_CA_FRACTAL (segments, n_thresholds, min_points)
%   opts.ca_break    which C-A break is the target threshold: 'highest' (default: the most
%                    anomalous population) | 'lowest' | k (k-th break, ascending)
%   opts.cell_area_km2  area of one cell, for the km^2 column (default 1)
%
%   Thresholds
%     P-A (Yousefi & Carranza 2015): the score t where the prediction rate (% deposits with
%         score >= t) equals 100 - % area with score >= t. At this point the map captures
%         P % of the deposits in (100 - P) % of the area, the best trade-off between the two.
%     C-A (Cheng et al. 1994): break between the fractal populations of the score, see
%         MPM_CA_FRACTAL; needs no deposits, so it also works for unexplored regions.
%
%   B fields
%     maps.pa / maps.ca   H x W single: 1 target, 0 non-target, NaN outside the study area
%     thresholds          struct(pa, ca)
%     table               one row per method: threshold, area (% and km^2), train / test
%                         deposits captured (% and n), efficiency = captured % / area %
%     ca                  C-A curve and fit (MPM_CA_FRACTAL output), pa (MPM_EVAL_MAP output)
if nargin < 5, opts = struct(); end
opts = mpm_merge(struct('methods', {{'pa', 'ca'}}, 'pa_deposits', 'train', 'ca', struct(), ...
    'ca_break', 'highest', 'cell_area_km2', 1), opts);
methods = cellstr(opts.methods);
s = double(scoreMap(valid));
trainLin = trainLin(:); testLin = testLin(:);
B = struct('maps', struct(), 'thresholds', struct(), 'table', table(), 'ca', [], 'pa', [], 'messages', {{}});
rows = {};
for m = methods(:)'
    switch lower(m{1})
        case 'pa'
            dep = trainLin; if strcmpi(opts.pa_deposits, 'all'), dep = [trainLin; testLin]; end
            if isempty(dep)
                B.messages{end+1} = 'P-A skipped: no deposits.'; continue;
            end
            bg = true(size(valid)); bg(dep) = false;
            C = mpm_eval_map('P-A', s, double(scoreMap(dep)), double(scoreMap(valid & bg)));
            B.pa = C;
            thr = C.pa_intersection.threshold;
            if ~isfinite(thr), B.messages{end+1} = 'P-A skipped: curves do not intersect.'; continue; end
            label = sprintf('P-A intersection (%.1f %% of known points / %.1f %% area)', ...
                C.pa_intersection.prediction, C.pa_intersection.area);
        case 'ca'
            CA = mpm_ca_fractal(s, mpm_merge(opts.ca, struct('cell_area', opts.cell_area_km2)));
            B.ca = CA;
            if isempty(CA.breaks), B.messages{end+1} = 'C-A skipped: no break found.'; continue; end
            if ischar(opts.ca_break) || isstring(opts.ca_break)
                if strcmpi(opts.ca_break, 'lowest'), k = 1; else, k = numel(CA.breaks); end
            else
                k = min(max(1, round(opts.ca_break)), numel(CA.breaks));
            end
            thr = CA.breaks(k);
            label = sprintf('C-A fractal break %d of %d (%d segments)', k, numel(CA.breaks), height(CA.segments));
        otherwise
            error('mpm:binary', 'Unknown binary target method "%s" (pa | ca).', m{1});
    end
    Bm = nan(size(valid), 'single');
    Bm(valid) = single(s >= thr);
    B.maps.(lower(m{1})) = Bm;
    B.thresholds.(lower(m{1})) = thr;
    area = mean(s >= thr) * 100;
    [pTr, nTr] = captured(Bm, trainLin); [pTe, nTe] = captured(Bm, testLin);
    rows(end+1, :) = {string(upper(m{1})), string(label), thr, area, area / 100 * numel(s) * opts.cell_area_km2, ...
        pTr, nTr, numel(trainLin), pTe, nTe, numel(testLin), pTr / area, pTe / area}; %#ok<AGROW>
end
if ~isempty(rows)
    B.table = cell2table(rows, 'VariableNames', {'method', 'rule', 'threshold', 'area_pct', 'area_km2', ...
        'train_captured_pct', 'train_captured', 'train_total', 'test_captured_pct', 'test_captured', ...
        'test_total', 'train_efficiency', 'test_efficiency'});
end
end

function [pct, n] = captured(Bm, lin)
if isempty(lin), pct = NaN; n = 0; return; end
v = Bm(lin); n = sum(v == 1);
pct = 100 * n / numel(lin);
end
