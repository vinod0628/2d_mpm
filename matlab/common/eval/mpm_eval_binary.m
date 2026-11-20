function E = mpm_eval_binary(name, y, s, threshold)
%MPM_EVAL_BINARY Sample-level classification metrics (ROC/AUC + threshold metrics).
%
%   E = mpm_eval_binary(name, y, s, threshold)
%   y logical (true = deposit), s score/probability, threshold for the confusion matrix.
%
%   E fields: name, n, npos, nneg, roc (see MPM_ROC), auc, threshold, confusion [TN FP; FN TP],
%             accuracy, precision, recall (sensitivity), specificity, f1, kappa, brier
y = logical(y(:)); s = double(s(:));
E.name = name;
E.n = numel(y); E.npos = sum(y); E.nneg = sum(~y);
E.roc = mpm_roc(y, s);
E.auc = E.roc.auc;
E.threshold = threshold;
yp = s >= threshold;
TP = sum(yp & y); TN = sum(~yp & ~y); FP = sum(yp & ~y); FN = sum(~yp & y);
E.confusion = [TN FP; FN TP];
E.accuracy = (TP + TN) / max(E.n, 1);
E.precision = TP / max(TP + FP, 1);
E.recall = TP / max(TP + FN, 1);
E.specificity = TN / max(TN + FP, 1);
E.f1 = 2 * TP / max(2 * TP + FP + FN, 1);
po = E.accuracy;
pe = ((TP + FP) * (TP + FN) + (TN + FN) * (TN + FP)) / max(E.n, 1)^2;
E.kappa = (po - pe) / max(1 - pe, eps);
if all(s >= 0 & s <= 1), E.brier = mean((s - y).^2); else, E.brier = NaN; end
end
