function M = clf_template()
%CLF_TEMPLATE Template for a new classification (prospectivity) method.
%
%   HOW TO ADD A METHOD (e.g. gradient boosting, MLP, RVM, logistic regression):
%     1. Copy this file to  classification/methods/clf_<key>.m  and rename the function.
%     2. Fill in key/name/defaults/grid and the train / predict local functions below.
%     3. Copy classification/run_svm_classification.m to run_<key>_classification.m and change
%        the line  M = clf_svm();  to  M = clf_<key>();
%   Nothing else changes: data loading, sampling, CV tuning, map prediction, ROC/AUC,
%   success-rate / P-A curves, target delineation, saving and the results GUI are shared.
%
%   CONTRACT (fields read by MPM_CLASSIFICATION_RUN)
%     key, name, task='classification', description
%     defaults              struct of hyper-parameters (cfg.params overrides single fields)
%     grid                  struct of candidate values for CV tuning; struct() = no tuning
%     score_is_probability  true if predict returns probabilities in [0,1] (threshold 0.5 valid)
%     has_uncertainty       false for classification (predict returns 1 output)
%     train(X, y, p, ctx)   X: n x P z-scored single, y: n x 1 logical (true = deposit),
%                           p: parameter struct, ctx.final: false inside CV, true for the final
%                           model (do expensive extras like calibration only then). Returns model.
%     predict(mdl, X)       n x 1 score, higher = more prospective
%     native_importance     [] or @(mdl, featureNames) -> table(feature, importance)
%     describe              [] or @(mdl, p) -> char summary shown in the GUI
%   Optional:
%     learning              'unsupervised' -> anomaly detector: train(Xcells, [], p, ctx) is fitted on
%                           cfg.anomaly.n_fit random UNLABELLED cells; deposits only tune / threshold
%                           / validate (see clf_iforest, clf_ocsvm, clf_autoencoder)
%                           'pu' -> positive-unlabelled learning: y = false marks UNLABELLED cells
%                           (cfg.pu.n_unlabelled random cells of the study area, no distance
%                           buffer) instead of background negatives (see clf_pubag)
%     input                 'patch' -> X passed to train / predict is the column of LINEAR CELL
%                           INDICES; the z-scored evidence cube is mpm_patch_source.active() (cut
%                           windows with src.extract(lin, w)); see clf_cnn
%     predict_map           @(mdl, F) -> [score, uncertainty] for all valid cells F.idx at once
%                           (e.g. fully convolutional evaluation) instead of chunked predict
%     spatial_reach         @(params, grid) -> cells; CV groups link samples within this
%                           Chebyshev distance (patch methods: windows that can overlap)
%     score_name            map label, e.g. 'anomaly score' (default 'prospectivity')
%     has_uncertainty, relevant_samples   see README
%
%   NOTE: local functions must not be called "predict" / "train" - they would shadow the
%   toolbox functions inside this file (hence fit_model / predict_score).
%
%   The example body below is an L2-regularised logistic regression (fitclinear), so a copy of
%   this file runs as-is.

M.key = 'template';
M.name = 'Template (logistic regression example)';
M.task = 'classification';
M.description = 'Skeleton for new methods; example body = ridge logistic regression.';
M.defaults = struct('Lambda', 1e-3);
M.grid = struct('Lambda', [1e-4 1e-3 1e-2 1e-1]);
M.score_is_probability = true;
M.has_uncertainty = false;
M.train = @fit_model;
M.predict = @predict_score;
M.native_importance = @native_importance;
M.describe = [];
end

function mdl = fit_model(X, y, p, ctx) %#ok<INUSD>
mdl = fitclinear(double(X), logical(y), 'Learner', 'logistic', 'Regularization', 'ridge', ...
    'Lambda', p.Lambda, 'ClassNames', [false; true]);
end

function s = predict_score(mdl, X)
[~, sc] = predict(mdl, double(X));
s = sc(:, 2);
end

function T = native_importance(mdl, names)
T = table(string(names(:)), abs(mdl.Beta(:)), 'VariableNames', {'feature', 'importance'});
T = sortrows(T, 'importance', 'descend');
T.Properties.Description = '|coefficient| on z-scored features';
end
