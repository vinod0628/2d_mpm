function M = reg_rf()
%REG_RF Random forest regression for grade estimation.
%
%   Model : TreeBagger regression trees. predict returns the mean of the trees and, as an
%           uncertainty map, the standard deviation of the tree predictions (model scale).
%
%   Hyper-parameters:
%     NumTrees / NumTreesTuning   trees of the final model / inside CV
%     MinLeafSize                 minimum observations per leaf (5 is the usual regression default)
%     NumPredictorsToSample       MULTIPLIER of P/3 (rounded, clipped to 1..P)
%     UseParallel                 use an open parallel pool

M.key = 'rf';
M.name = 'Random forest regression';
M.task = 'regression';
M.description = 'Bagged regression trees (TreeBagger); tree spread as uncertainty.';
M.defaults = struct('NumTrees', 300, 'NumTreesTuning', 100, 'MinLeafSize', 5, ...
    'NumPredictorsToSample', 1, 'UseParallel', false);
M.grid = struct('MinLeafSize', [1 3 5 10], 'NumPredictorsToSample', [0.5 1 2]);
M.has_uncertainty = true;
M.train = @fit_model;
M.predict = @predict_value;
M.native_importance = @native_importance;
M.describe = @describe;
end

function mdl = fit_model(X, y, p, ctx)
P = size(X, 2);
m = max(1, min(P, round(p.NumPredictorsToSample * P / 3)));
nt = p.NumTrees; if ~ctx.final, nt = p.NumTreesTuning; end
args = {'Method', 'regression', 'MinLeafSize', p.MinLeafSize, 'NumPredictorsToSample', m, ...
    'Options', statset('UseParallel', p.UseParallel)};
if ctx.final, args = [args, {'OOBPrediction', 'on', 'OOBPredictorImportance', 'on'}]; end
mdl = TreeBagger(nt, double(X), double(y), args{:});
if ~ctx.final, mdl = compact(mdl); end
end

function [y, sd] = predict_value(mdl, X)
[y, sd] = predict(mdl, double(X));
end

function T = native_importance(mdl, names)
T = table();
if isprop(mdl, 'OOBPermutedPredictorDeltaError') && ~isempty(mdl.OOBPermutedPredictorDeltaError)
    T = table(string(names(:)), mdl.OOBPermutedPredictorDeltaError(:), 'VariableNames', {'feature', 'importance'});
    T = sortrows(T, 'importance', 'descend');
    T.Properties.Description = 'RF out-of-bag permuted predictor delta error';
end
end

function txt = describe(mdl, p)
txt = sprintf('%d trees, MinLeafSize = %d, NumPredictorsToSample = %d', mdl.NumTrees, p.MinLeafSize, ...
    mdl.NumPredictorsToSample);
end
