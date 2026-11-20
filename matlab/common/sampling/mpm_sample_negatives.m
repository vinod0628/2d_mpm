function [negTrain, negTest, distMap] = mpm_sample_negatives(F, R, depositLin, nTrain, nTest, opts, rs)
%MPM_SAMPLE_NEGATIVES Draw background (non-deposit) cells for classifier training and testing.
%
%   [negTrain, negTest] = mpm_sample_negatives(F, R, depositLin, nTrain, nTest, opts, rs)
%
%   depositLin   linear indices of ALL known deposit cells (train + test): no negative is drawn
%                closer than opts.min_distance to any of them
%   opts.strategy      'random' (default): uniform among valid cells beyond min_distance
%   opts.min_distance  map units (m)
%   Returns linear cell indices; train and test negatives never overlap.
%
%   New strategies (e.g. low-favourability or target-pattern sampling) can be added as cases here.

if nargin < 6, opts = struct(); end
opts = mpm_merge(struct('strategy', 'random', 'min_distance', 0), opts);

dep = false(R.RasterSize);
dep(depositLin) = true;
distMap = bwdist(dep) * R.CellExtentInWorldX;          % metres to nearest deposit cell
cand = F.idx(distMap(F.idx) > max(opts.min_distance, 0) & ~dep(F.idx));

switch lower(opts.strategy)
    case 'random'
        need = nTrain + nTest;
        if numel(cand) < need
            error('mpm:negatives', 'Only %d candidate background cells for %d negatives; reduce min_distance/ratio.', ...
                numel(cand), need);
        end
        pick = cand(randperm(rs, numel(cand), need));
    otherwise
        error('mpm:negatives', 'Unknown negative-sampling strategy "%s".', opts.strategy);
end
negTrain = pick(1:nTrain);
negTest = pick(nTrain+1:end);
end
