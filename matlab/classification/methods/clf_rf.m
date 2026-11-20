function M = clf_rf()
%CLF_RF Random forest classifier for prospectivity mapping (Breiman 2001; Carranza & Laborte 2015).
%
%   Model : TreeBagger (Statistics and Machine Learning Toolbox), classification trees.
%   Output: fraction of trees voting "deposit" (probability-like, 0..1).
%
%   Hyper-parameters:
%     NumTrees              trees of the final model
%     NumTreesTuning        trees used inside CV tuning (faster)
%     MinLeafSize           minimum observations per leaf
%     NumPredictorsToSample MULTIPLIER of sqrt(P) (rounded, clipped to 1..P)
%     BalanceClasses        true -> uniform class prior
%     UseParallel           true -> use a parallel pool if one is open
%   Native importance: out-of-bag permuted predictor delta error (per feature column).

M.key = 'rf';
M.name = 'Random forest';
M.task = 'classification';
M.description = 'Bagged classification trees (TreeBagger) with OOB importance.';
M.defaults = struct('NumTrees', 300, 'NumTreesTuning', 100, 'MinLeafSize', 1, ...
    'NumPredictorsToSample', 1, 'BalanceClasses', true, 'UseParallel', false);
M.grid = struct('MinLeafSize', [1 3 5], 'NumPredictorsToSample', [0.5 1 2]);
M.score_is_probability = true;
M.has_uncertainty = false;
M.train = @fit_model;
M.predict = @predict_score;
M.native_importance = @native_importance;
M.describe = @describe;
end

function mdl = fit_model(X, y, p, ctx)
P = size(X, 2);
m = max(1, min(P, round(p.NumPredictorsToSample * sqrt(P))));
nt = p.NumTrees; if ~ctx.final, nt = p.NumTreesTuning; end
args = {'Method', 'classification', 'MinLeafSize', p.MinLeafSize, 'NumPredictorsToSample', m, ...
    'Options', statset('UseParallel', p.UseParallel)};
if ctx.final, args = [args, {'OOBPrediction', 'on', 'OOBPredictorImportance', 'on'}]; end
if p.BalanceClasses, args = [args, {'Prior', 'uniform'}]; end
mdl = TreeBagger(nt, double(X), double(y), args{:});
if ~ctx.final, mdl = compact(mdl); end
end

function s = predict_score(mdl, X)
[~, sc] = predict(mdl, double(X));
s = sc(:, strcmp(mdl.ClassNames, '1'));
end

function T = native_importance(mdl, names)
T = table();
if isprop(mdl, 'OOBPermutedPredictorDeltaError') && ~isempty(mdl.OOBPermutedPredictorDeltaError)
    T = table(string(names(:)), mdl.OOBPermutedPredictorDeltaError(:), ...
        'VariableNames', {'feature', 'importance'});
    T = sortrows(T, 'importance', 'descend');
    T.Properties.Description = 'RF out-of-bag permuted predictor delta error';
end
end

function txt = describe(mdl, p)
txt = sprintf('%d trees, MinLeafSize = %d, NumPredictorsToSample = %d', mdl.NumTrees, ...
    p.MinLeafSize, mdl.NumPredictorsToSample);
end
