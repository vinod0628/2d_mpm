function T = mpm_metrics_table(res)
%MPM_METRICS_TABLE One-row-per-set summary of a classification or regression result.
switch res.task
    case 'classification'
        rows = {};
        sets = {'train', 'test'};
        for i = 1:2
            if isfield(res.eval.samples, sets{i})
                E = res.eval.samples.(sets{i});
                rows(end+1, :) = {string(E.name), E.npos, E.nneg, E.auc, NaN, E.threshold, E.accuracy, ...
                    E.precision, E.recall, E.specificity, E.f1, E.kappa, E.brier, NaN, NaN, NaN}; %#ok<AGROW>
            end
        end
        for i = 1:2
            if isfield(res.eval.map, sets{i})
                C = res.eval.map.(sets{i});
                rows(end+1, :) = {string([C.name ' vs all cells']), C.npos, C.nneg, C.auc, C.sr_auc, NaN, ...
                    NaN, NaN, NaN, NaN, NaN, NaN, NaN, C.pa_intersection.threshold, ...
                    C.pa_intersection.prediction, C.pa_intersection.area}; %#ok<AGROW>
            end
        end
        T = cell2table(rows, 'VariableNames', {'set', 'positives', 'negatives', 'roc_auc', ...
            'success_rate_auc', 'threshold', 'accuracy', 'precision', 'recall', 'specificity', 'f1', ...
            'kappa', 'brier', 'pa_threshold', 'pa_prediction_pct', 'pa_area_pct'});
    case 'regression'
        rows = {};
        for sc = ["model", "grade"]
            for set = ["train", "test", "cv"]
                key = char(set + "_" + sc);
                if isfield(res.eval, key) && ~isempty(res.eval.(key))
                    E = res.eval.(key);
                    rows(end+1, :) = {string(E.name), E.n, E.r2, E.rmse, E.mae, E.bias, E.pearson, E.spearman}; %#ok<AGROW>
                end
            end
        end
        T = cell2table(rows, 'VariableNames', {'set', 'n', 'r2', 'rmse', 'mae', 'bias', 'pearson', 'spearman'});
        if isfield(res.eval, 'cutoff') && ~isempty(res.eval.cutoff)
            c = res.eval.cutoff;
            T.cutoff_auc = nan(height(T), 1);
            for i = 1:height(T)
                if startsWith(T.set(i), 'training ('), T.cutoff_auc(i) = c.train.auc; end
                if startsWith(T.set(i), 'test (') && isfield(c, 'test'), T.cutoff_auc(i) = c.test.auc; end
            end
        end
end
end
