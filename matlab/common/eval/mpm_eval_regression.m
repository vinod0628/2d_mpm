function E = mpm_eval_regression(name, yObs, yPred)
%MPM_EVAL_REGRESSION Regression metrics for grade estimation.
%
%   E = mpm_eval_regression(name, yObs, yPred)
%   E fields: name, n, r2 (coefficient of determination), rmse, mae, bias (mean pred - obs),
%             pearson, spearman, obs, pred, resid (= obs - pred)
yObs = double(yObs(:)); yPred = double(yPred(:));
ok = isfinite(yObs) & isfinite(yPred);
yObs = yObs(ok); yPred = yPred(ok);
E.name = name; E.n = numel(yObs);
r = yObs - yPred;
E.rmse = sqrt(mean(r.^2));
E.mae = mean(abs(r));
E.bias = mean(yPred - yObs);
sst = sum((yObs - mean(yObs)).^2);
E.r2 = 1 - sum(r.^2) / max(sst, eps);
if E.n > 2
    E.pearson = corr(yObs, yPred, 'Type', 'Pearson');
    E.spearman = corr(yObs, yPred, 'Type', 'Spearman');
else
    E.pearson = NaN; E.spearman = NaN;
end
E.obs = yObs; E.pred = yPred; E.resid = r;
end
