function M = clf_iforest()
%CLF_IFOREST Isolation forest anomaly detection for prospectivity mapping (unsupervised).
%
%   Idea (Chen & Wu 2019): mineralisation is a rare, local event, so the evidence signature of
%   mineralised ground is a multivariate ANOMALY of the study area. An isolation forest isolates
%   observations by random axis-parallel splits; anomalies are "few and different", so they are
%   isolated close to the root (short path length) - no density or distance computations.
%
%   Fit      : on cfg.anomaly.n_fit random cells of the study area, WITHOUT labels (deposits play
%              no part in the fit). Known deposits are used only to choose SampleSize (AUC of
%              training deposits vs. training background), the threshold (Youden index of the
%              training ROC, as in Chen & Wu) and for validation; test deposits stay independent.
%   Score    : s(x) = 2^(-E(h(x)) / c(psi)) in (0, 1] (Liu et al. 2008): s -> 1 anomaly,
%              s << 0.5 normal. Higher = more anomalous = more prospective.
%   Algorithm: written from the paper (Algorithms 1-3, Eqs. 1-2) in common/model/mpm_iforest_*.m,
%              including the height limit l = ceiling(log2 psi). MATLAB's iforest grows trees
%              until every point is isolated (no height limit) and is used only as a cross-check.
%
%   Hyper-parameters:
%     NumTrees     t (paper default 100; path lengths converge well before)
%     SampleSize   psi (paper default 256); tuned over the grid
%     HeightLimit  'auto' = ceiling(log2 psi) (paper) | 'full' | number
%
%   References: Liu, F.T., Ting, K.M., Zhou, Z.-H., 2008. Isolation forest. Proc. IEEE ICDM,
%   413-422.  Chen, Y., Wu, W., 2019. Isolation forest as an alternative data-driven mineral
%   prospectivity mapping method with a higher data-processing efficiency. Nat. Resour. Res. 28,
%   31-46.

M.key = 'iforest';
M.name = 'Isolation forest';
M.task = 'classification';
M.learning = 'unsupervised';
M.description = 'Unsupervised anomaly detection: isolation forest (own implementation of Liu et al. 2008).';
M.defaults = struct('NumTrees', 100, 'SampleSize', 256, 'HeightLimit', 'auto');
M.grid = struct('SampleSize', [64 128 256 512]);
M.score_is_probability = false;
M.score_name = 'anomaly score';
M.has_uncertainty = false;
M.train = @fit_model;
M.predict = @predict_score;
M.native_importance = [];
M.describe = @describe;
end

function mdl = fit_model(X, ~, p, ~)
mdl = mpm_iforest_train(X, struct('n_trees', p.NumTrees, 'sample_size', p.SampleSize, ...
    'height_limit', p.HeightLimit));
end

function s = predict_score(mdl, X)
s = mpm_iforest_score(mdl, X);
end

function txt = describe(mdl, p) %#ok<INUSD>
txt = sprintf('%d isolation trees, psi = %d, height limit %d (mean max depth %.1f), c(psi) = %.3f', ...
    mdl.n_trees, mdl.psi, mdl.height_limit, mdl.mean_depth, mdl.c_psi);
end
